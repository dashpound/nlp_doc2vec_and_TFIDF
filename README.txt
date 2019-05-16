About:

crawler -> Scrapy Crawler containing the Economics crawler, as well as the random article crawler

economics directory-> Contains the code leveraged to produce TF-IDF and Doc2Vec analysis
	run_economics.py -> actual .py file that produces outputs stored in "results" directory.
	run_economics_input.txt -> confermation of inputs in text format
	run_economics_output.txt -> terminal output in text format
	
	results directory -> contains doc2vec matricies & tfidf-matrices
	
	econ directory -> contains the aggregated and raw corpus files
		files directory -> combined random + econ corpora 
			all.jl -> master corpus (random + economcis)
		output_from_scrapy -> raw output from scrapy "items" files
			econ.jl -> items file from scrapy run
			random.jl -> items file from scrapy run
	