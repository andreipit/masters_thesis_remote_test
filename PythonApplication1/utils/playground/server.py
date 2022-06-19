
def built_in_server():
    # Desktop/tmp/socket/miniserver.py
    import http.server
    import socketserver
    PORT = 4567
    Handler = http.server.SimpleHTTPRequestHandler
    #with socketserver.TCPServer(("", PORT), Handler) as httpd:
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()
    # http://localhost:1234/ => Congratulations! The HTTP Server is working!


#import io
#class TextIOWrapper_noclose(io.TextIOWrapper):
#    def close(self):
#        if not self.closed:
#            self.closed = True
#            self.flush()
#            self.detach()

import http.server
class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):
        path = self.translate_path(self.path) # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1/
        #self.send_response(201, "Created")
        
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        #self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write("Thanks!\n".encode())
        #self.wfile.write(TextIOWrapper_noclose("Thanks!\n".encode()))

        ##if not self.wfile.closed:
        #with open(self.wfile, 'w') as f:
        #    f.write("Thanks".encode())
        #    #self.wfile.write("Thanks".encode())


        #json_line = '{"1":"11","2":"22"}'
        #coding=utf8

        #json_line = {"1":"11","2":"22"}
        #self.wfile.write(json_line)


        #self.wfile.close()

        #if path.endswith('/'):
        #    self.send_response(405, "Method Not Allowed")
        #    self.wfile.write("PUT not allowed on a directory\n".encode())
        #    return
        #else:
        #    try:
        #        os.makedirs(os.path.dirname(path))
        #    except FileExistsError: pass
        #    length = int(self.headers['Content-Length'])
        #    with open(path, 'wb') as f:
        #        f.write(self.rfile.read(length))
        #    self.send_response(201, "Created")

def built_in_with_put():
    import argparse
    import http.server
    import os
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--bind', '-b', default='', metavar='ADDRESS',
    #                    help='Specify alternate bind address '
    #                            '[default: all interfaces]')
    #parser.add_argument('port', action='store',
    #                    default=8000, type=int,
    #                    nargs='?',
    #                    help='Specify alternate port [default: 8000]')
    #args = parser.parse_args()

    #http.server.test(HandlerClass=HTTPRequestHandler, port=args.port, bind=args.bind)
    http.server.test(HandlerClass=HTTPRequestHandler, port=4567, bind='127.0.0.1')

def send_old_style():
    import socket
    import os
    # Standard socket stuff:
    host = ''
    #host = 'localhost'
    port = 1234
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5) 
    # Loop forever, listening for requests:
    while True:
        print('writing')
        csock, caddr = sock.accept()
        print("Connection from: " + str(caddr))
        req = csock.recv(1024)  # get the request, 1kB max
        print(req)
        # Look in the first line of the request for a move command
        # A move command should be e.g. 'http://server/move?a=90'
        filename = 'utils\playground\index.html' #'index.html' # static/index.html
        f = open(filename, 'r')
        csock.sendall(str.encode("HTTP/1.0 200 OK\n",'iso-8859-1'))
        csock.sendall(str.encode('Content-Type: text/html\n', 'iso-8859-1'))
        csock.send(str.encode('\r\n'))
        # send data per line
        for l in f.readlines():
            print('Sent ', repr(l))
            csock.sendall(str.encode(""+l+"", 'iso-8859-1'))
            l = f.read(1024)
        f.close()
        csock.close()
    # http://localhost:1234/ => Congratulations! The HTTP Server is working!

if __name__ == '__main__':
    #send_old_style()
    #built_in_server()
    built_in_with_put()

