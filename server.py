#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import urlparse, parse_qs
import autocomplete
import pdb

# HTTPRequestHandler class
# Pieces found online for general skeleton of run function and class
# https://daanlenaerts.com/blog/2015/06/03/create-a-simple-http-server-with-python-3/
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):

        self.send_response(200)
        # Send headers
        self.send_header('Content-type','application/json')
        self.end_headers()

        #Parse query
        query = parse_qs(urlparse(self.path).query)
        # print(query)

        self.wfile.write(json.dumps(autocomplete.autocomplete(trie, query['q'][0], 3)).encode('utf8'))

def run():
  print('starting server...')

  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
  server_address = ('127.0.0.1', 8081)
  httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)

  print('running server...')
  httpd.serve_forever()

trie = autocomplete.process_data()
run()
