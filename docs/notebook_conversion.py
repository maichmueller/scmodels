import os

import nbformat
from traitlets.config import Config
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor, ExecutePreprocessor

# Setup config
c = Config()

c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)
c.TagRemovePreprocessor.remove_input_tags = ("remove_input",)
c.TagRemovePreprocessor.enabled = True

current_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(current_dir, "README.ipynb")
with open(filepath, "r") as file:
    nb = nbformat.read(file, as_version=nbformat.NO_CONVERT)

ep = ExecutePreprocessor(timeout=600, kernel_name='python')
ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})


exporter = MarkdownExporter(config=c)
exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)
output, metadata = exporter.from_notebook_node(nb)


# Write to output markdown file
with open(os.path.abspath(os.path.join(current_dir, '..', "README.md")), "w") as f:
    f.write(output)
