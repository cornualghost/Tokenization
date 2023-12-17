import morfessor

# Morfessor pipeline
class MorfessorTokenizationPipeline:
    def __init__(self, model_path):
        self.model = morfessor.MorfessorIO().read_binary_model_file(model_path)

    def tokenize(self, text):
        return self.model.viterbi_segment(text)[0]