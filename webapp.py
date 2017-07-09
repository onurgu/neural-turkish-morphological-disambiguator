# coding=utf-8
import tornado.ioloop
import tornado.web

from train import create_parser, create_model_for_disambiguation, disambiguate_single_line_sentence

parser = create_parser()

parser.add_argument("--port", default=8888, type=int)

args = parser.parse_args()


class DisambiguationHandler(tornado.web.RequestHandler):

    model = None

    def initialize(self):

        if DisambiguationHandler.model is None:
            label2ids, params, model = create_model_for_disambiguation(args)

            DisambiguationHandler.label2ids = label2ids
            DisambiguationHandler.params = params
            DisambiguationHandler.model = model

    def disambiguate_line(self, line):
        analyzer_output_string, prediction_lines, prediction_lines_raw = disambiguate_single_line_sentence(line,
                                                 DisambiguationHandler.model,
                                                 DisambiguationHandler.label2ids,
                                                 DisambiguationHandler.params,
                                          print_prediction_lines=False)

        return {
            'analyzer_output': {i: line.decode('iso-8859-9') for i, line in enumerate(analyzer_output_string.split("\n")) if len(line) > 0},
            'disambiguator_output': {i: {'surface_form': surface_form, 'analysis': analysis} for i, (surface_form, analysis) in enumerate(prediction_lines_raw)}}


    @tornado.web.asynchronous
    def post(self):

        self.add_header("Access-Control-Allow-Origin", "*")
        line = self.get_argument("single_line_sentence", default=u"Dünyaya hoş geldiniz.")
        # print type(line)
        print(line)

        line = line.strip()
        self.write(self.disambiguate_line(line))
        self.write("\n")
        self.finish()


def make_app():
    return tornado.web.Application([
        (r"/disambiguate/", DisambiguationHandler),
    ])


def start_webapp():

    print("Creating app object")
    app = make_app()
    print("Listening")
    if args.port:
        app.listen(args.port)
    else:
        app.listen(8888)
    print("Starting the loop")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    print("Starting webapp")
    start_webapp()