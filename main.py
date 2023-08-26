import wafflecone

wafflecone.version()

male_seed = ["male", "he", "boy"]
female_seed = ["female", "she", "girl"]

exclude=["[CLS]", "[SEP]"]

calculator = wafflecone.calculator(path="./test_data/layer12.json",
                                   subspace_seeds=[male_seed, female_seed],
                                   exclude_words=exclude,
                                   user_friendly=True)
print(calculator.bias_sum_average())
print(calculator.bias_asb_sum_average())
calculator.save_summary()

wafflecone.visualize(3000)
