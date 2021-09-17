"""
 ////////////////////////////////////////////////////////////////////////////
 //
 // Copyright (C) NVIDIA Corporation.  All rights reserved.
 //
 // NVIDIA Sample Code
 //
 // Please refer to the NVIDIA end user license agreement (EULA) associated
 // with this source code for terms and conditions that govern your use of
 // this software. Any use, reproduction, disclosure, or distribution of
 // this software and related documentation outside the terms of the EULA
 // is strictly prohibited.
 //
 ////////////////////////////////////////////////////////////////////////////
"""
import cherrypy
import pathlib
import io
from base64 import b64encode
from models_infer import Model
from wait_socket import wait_for_port
wait_for_port(50051, "riva", 120)


m = Model()


WEB_ROOT = str(pathlib.Path(__file__).parent.absolute())+'/client'
print(WEB_ROOT)


def stop_clean():
    print('stopped')


def run_server():

    cherrypy.config.update({
        'server.socket_port': 8888,
        #        'environment': 'production',
        'engine.autoreload.on': False,
        #        'server.thread_pool':  1,
        'server.socket_host': '0.0.0.0',
        'tools.staticdir.on': True,
        'tools.staticdir.dir': WEB_ROOT,
        'tools.staticdir.index': 'index.html'
    })

    cherrypy.server.ssl_certificate = "cert.pem"
    cherrypy.server.ssl_private_key = "privkey.pem"

    class HelloWorld(object):

        @cherrypy.expose
        def doc(self):
            p = pathlib.Path('text/doc.txt')
            if p.exists():
                with io.open(str(p), 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            else:
                return ""

        @cherrypy.expose
        @cherrypy.tools.json_out()
        def questions(self):
            p = pathlib.Path('text/questions.txt')
            if p.exists():
                with io.open(str(p), 'r', encoding='utf-8') as f:
                    content = f.readlines()
                return content
            else:
                return []

        @cherrypy.expose
        @cherrypy.tools.json_out()
        @cherrypy.tools.json_in()
        def infer(self):
            input_json = cherrypy.request.json
            r = m.qa_infer(input_json['para'], input_json['question'])
            return [r]

        @cherrypy.expose
        def asr(self, audio_data):
            inputs = audio_data
            r = m.asr_infer(inputs.file)
            return r

        @cherrypy.expose
        @cherrypy.tools.json_in()
        def tacotron(self):
            input_json = cherrypy.request.json
            r = m.tacotron_infer(input_json['text'])
            print('input', input_json['text'])
            cherrypy.response.headers[
                'Content-Type'] = 'application/octet-stream'
            return b64encode(r)

    cherrypy.engine.subscribe('stop', stop_clean)
    cherrypy.quickstart(HelloWorld())


if __name__ == '__main__':
    run_server()
