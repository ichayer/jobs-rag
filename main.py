from pdfminer.high_level import extract_text
from json_helper import InputData as input

text = extract_text("cvs/CV-DESARROLLADOR.pdf")

llm = input.llm()

data = llm.invoke(input.input_data(text))
print(data)