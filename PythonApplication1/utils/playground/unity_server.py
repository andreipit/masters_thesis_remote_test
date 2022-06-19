import http.server

class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):
        path: str = self.translate_path(self.path) # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1/
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("My mes!\n".encode())
        length: int = int(self.headers['Content-Length'])
        NewData: str = self.rfile.read(length)
        #print('in', self.request)
        print('REQUEST==',NewData)

def built_in_with_put():
    http.server.test(HandlerClass=HTTPRequestHandler, port=4567, bind='127.0.0.1')

if __name__ == '__main__':
    built_in_with_put()

