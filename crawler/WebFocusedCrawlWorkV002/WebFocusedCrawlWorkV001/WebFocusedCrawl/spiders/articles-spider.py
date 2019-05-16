# Scrape Wikipedia, saving page html code to wikipages directory 
# Most Wikipedia pages have lots of text 
# We scrape the text data creating a JSON lines file items.jl
# with each line of JSON representing a Wikipedia page/document
# Subsequent text parsing of these JSON data will be needed
# This example is for the topic robotics
# Replace the urls list with appropriate Wikipedia URLs
# for your topic(s) of interest

# ensure that NLTK has been installed along with the stopwords corpora
# pip install nltk
# python -m nltk.downloader stopwords

import scrapy
import os.path
from WebFocusedCrawl.items import WebfocusedcrawlItem  # item class 
import nltk  # used to remove stopwords from tags
import re  # regular expressions used to parse tags

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    good_tokens = [token for token in tokens if token not in stopword_list]
    return good_tokens    
 
sor = "https://en.wikipedia.org/wiki/"

class ArticlesSpider(scrapy.Spider):
    name = "articles-spider"

    def start_requests(self):
        allowed_domains = ['en.wikipedia.org']

        # list of Wikipedia URLs for topic of interest
        urls = ["Economics",
                "Microeconomics",
                "Macroeconomics",
                "Economy",
                "Money"
                "Fiscal_policy",
                "Monetary_policy",
                "Central_bank",
                "gross_domestic_product",
                "inflation_rates",
                "Janet_Yellen",
                "Federal_Reserve",
                "Federal_Reserve_Bank",
                "Ben_Bernake",
                "Economist",
                "Jerome_Powell",
                "Gini_Index",
                "Currency",
                "Gold_Standard",
                "Central_Bank",
                "Bitcoin",
                "Cryptocurrency",
                "Exchange_Rates",
                "International_Monetary_Fund",
                "John_Maynard_Keynes", 
                "Economic_Indicators",
                "Leading_Indicators",
                "Lagging_Indicators",
                "Interest_Rates",
                "Expanding_Economy",
                "Business_Cycles", 
                "Paul_Krugman", 
                "Adam_Smith", 
                "Capitalism",
                "Socialism",
                "Communism",
                "The_Wealth_of_Nations",
                "Classical_economics",
                "division_of_labour",
                "free_markets",
                "invisible_hand",
                "Milton_Friedman",
                "Alan_Greenspan",
                "Depression",
                "Recession",
                "Taxation",
                "stagflation",
                "Reserve_Currency",
                "Unemployment",
                "Supply",
                "Demand",
                "Irrational_Exuberance",
                "Stock_Market",
                "Free_Trade",
                "Finance",
                "Banks",
                "Insurance",
                "Option_(finance)",
                "Derivative_(finance)",
                "Investors"
                "consumption",
                "Savings",
                "Security_(finance)",
                "Accounting",
                "Investment",
                "government_debt",
                "World_Bank",
                "Bonds",
                "Financial_Instruments",
                "Cash",
                "Exchange",
                "Trade",
                "Balance_Sheet",
                "Wealth",
                "Taxes",
                "Progressive_Tax",
                "Regressive_tax",
                "Financial_crisis",
                "Financial_markets",
                "Great_Depression",
                "Hyperinflation", 
                "Scarcity",
                "Human_Capital",
                "Externality",
                "Purchasing_Power_Parity",
                "Physical_Capital",
                "Social_Capital",
                "Natural_Capital",
                "Equilibrium",
                "Competition",
                "The_Dismal_science",
                "Developed_country",
                "Real_versus_nominal_value_(economics)",
                "Price_index",
                "Game_Theory",
                "Debasement",
                "Neoclassical_economics",
                "Austerity",
                "Deficit_spending",
                "Crowding_out_(economics)",
                "Dirigisme",
                "Mercantilism",
                "World_Trade_Organization",
                "Austrian_School",
                "Schools_of_economic_thought",
                "Chicago_school_of_economics",
                "Nominal_rigidity",
                "Involuntary_unemployment",
                "Shapiro–Stiglitz_theory",
                "Nobel_Memorial_Prize_in_Economic_Sciences",
                "Joseph_Stiglitz",
                "Globalization",
                "Tariff",
                "Poverty",
                "Debt_of_developing_countries",
                "Debt",
                "OPEC",
                "Monopoly",
                "Market_(economics)",
                "Free_market",
                "Information_asymmetry"
                ]
            
        for url in urls:
            yield scrapy.Request(url=sor+url, callback=self.parse)

    def parse(self, response):
        # first part: save wikipedia page html to wikipages directory
        page = response.url.split("/")[4]
        page_dirname = 'wikipages'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename) 

        # second part: extract text for the item for document corpus
        item = WebfocusedcrawlItem()
        item['url'] = response.url
        item['labels'] = 'economics'
        item['title'] = response.css('h1::text').extract_first()
        item['text'] = response.xpath('//div[@id="mw-content-text"]//text()')\
                           .extract()                                                             
        tags_list = [response.url.split("/")[2],
                     response.url.split("/")[3]]
        more_tags = [x.lower() for x in remove_stopwords(response.url\
                       	    .split("/")[4].split("_"))]
        for tag in more_tags:
            tag = re.sub('[^a-zA-Z]', '', tag)  # alphanumeric values only  
            tags_list.append(tag)
        item['tags'] = tags_list                       
        return item 
 
