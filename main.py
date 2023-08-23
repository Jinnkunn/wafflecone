import wafflecone

wafflecone.version()

calculator = wafflecone.calculator("./test_data/layer12.json")
print(calculator.bias_sum_average())
print(calculator.bias_asb_sum_average())