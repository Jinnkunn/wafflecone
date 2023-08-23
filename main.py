import wafflecone

wafflecone.version()

male_seed = ["male", "he", "boy"]
female_seed = ["female", "she", "girl"]

calculator = wafflecone.calculator("./test_data/layer12.json", [male_seed, female_seed])
print(calculator.bias_sum_average())
print(calculator.bias_asb_sum_average())
calculator.save_summary()