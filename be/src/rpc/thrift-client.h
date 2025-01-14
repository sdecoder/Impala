// Copyright 2012 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef IMPALA_RPC_THRIFT_CLIENT_H
#define IMPALA_RPC_THRIFT_CLIENT_H

#include <ostream>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <common/status.h>
#include <thrift/Thrift.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TSSLSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <sstream>
#include <gflags/gflags.h>

#include "transport/TSaslClientTransport.h"
#include "transport/TSasl.h"
#include "rpc/authentication.h"
#include "rpc/thrift-server.h"
#include "gen-cpp/Types_types.h"

DECLARE_string(principal);
DECLARE_string(hostname);

namespace impala {
/// Super class for templatized thrift clients.
class ThriftClientImpl {
 public:
  ~ThriftClientImpl() {
    Close();
  }

  const TNetworkAddress& address() const { return address_; }

  /// Open the connection to the remote server. May be called repeatedly, is idempotent
  /// unless there is a failure to connect.
  Status Open();

  /// Retry the Open num_retries time waiting wait_ms milliseconds between retries.
  /// If num_retries == 0, the connection is retried indefinitely.
  Status OpenWithRetry(uint32_t num_retries, uint64_t wait_ms);

  /// Close the connection with the remote server. May be called repeatedly.
  void Close();

  /// Set receive timeout on the underlying TSocket.
  void setRecvTimeout(int32_t ms) { socket_->setRecvTimeout(ms); }

  /// Set send timeout on the underlying TSocket.
  void setSendTimeout(int32_t ms) { socket_->setSendTimeout(ms); }

 protected:
  ThriftClientImpl(const std::string& ipaddress, int port, bool ssl)
      : address_(MakeNetworkAddress(ipaddress, port)), ssl_(ssl) {
    socket_create_status_ = CreateSocket();
  }

  /// Create a new socket without opening it. Returns an error if the socket could not
  /// be created.
  Status CreateSocket();

  /// Address of the server this client communicates with.
  TNetworkAddress address_;

  /// True if ssl encryption is enabled on this connection.
  bool ssl_;

  Status socket_create_status_;

  /// Sasl Client object.  Contains client kerberos identification data.
  /// Will be NULL if kerberos is not being used.
  boost::shared_ptr<sasl::TSasl> sasl_client_;

  /// All shared pointers, because Thrift requires them to be
  boost::shared_ptr<apache::thrift::transport::TSocket> socket_;
  boost::shared_ptr<apache::thrift::transport::TTransport> transport_;
  boost::shared_ptr<apache::thrift::protocol::TBinaryProtocol> protocol_;
};


/// Utility client to a Thrift server. The parameter type is the Thrift interface type
/// that the server implements.
/// TODO: Consider a builder class to make constructing this class easier.
template <class InterfaceType>
class ThriftClient : public ThriftClientImpl {
 public:
  /// Creates, but does not connect, a new ThriftClient for a remote server.
  ///  - ipaddress: address of remote server
  ///  - port: port on which remote service runs
  ///  - service_name: If set, the target service to connect to.
  ///  - auth_provider: Authentication scheme to use. If NULL, use the global default
  ///    client<->demon authentication scheme.
  ///  - ssl: if true, SSL is enabled on this connection
  ThriftClient(const std::string& ipaddress, int port,
      const std::string& service_name = "", AuthProvider* auth_provider = NULL,
      bool ssl = false);

  /// Returns the object used to actually make RPCs against the remote server
  InterfaceType* iface() { return iface_.get(); }

 private:
  boost::shared_ptr<InterfaceType> iface_;

  AuthProvider* auth_provider_;
};

template <class InterfaceType>
ThriftClient<InterfaceType>::ThriftClient(const std::string& ipaddress, int port,
    const std::string& service_name,
    AuthProvider* auth_provider, bool ssl)
    : ThriftClientImpl(ipaddress, port, ssl),
      iface_(new InterfaceType(protocol_)),
      auth_provider_(auth_provider) {
  // Below is one line of code in ThriftClientImpl::Close(),
  // if (transport_.get != NULL && transport_->isOpen()) transport_->close();
  // Here transport_->isOpen() will call socker_->isOpen(), when socket_ is NULL,
  // it will crash
  if (socket_ != NULL) {
    ThriftServer::BufferedTransportFactory factory;
    transport_ = factory.getTransport(socket_);
  }

  if (auth_provider_ == NULL) {
    auth_provider_ = AuthManager::GetInstance()->GetInternalAuthProvider();
  }

  auth_provider_->WrapClientTransport(address_.hostname, transport_, service_name,
      &transport_);

  protocol_.reset(new apache::thrift::protocol::TBinaryProtocol(transport_));
  iface_.reset(new InterfaceType(protocol_));
}

}
#endif
