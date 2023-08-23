import wafflecone

wafflecone.version()

result = wafflecone.bias_asb_sum_average("./test_data/layer12.json")
print(result)