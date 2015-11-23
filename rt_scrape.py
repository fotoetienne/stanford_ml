#!/usr/bin/env python2.7
"""
# rtscrape.py #
Scrape recent house sales data from realtytrac.com
 - csv output "<sqft>, <price>"
 - fast and lazy w/ async requests and generater exprns
 - specify query and number of results pages to parse
 - each page has <= 50 results
 - some results lack sqft and are discarded

## Usage  ##
rt_scrape.py <query> <n-pages> > <output.csv>

## Example ##
$ ./rt_scrape.py "tn/hamilton-county/chattanooga" 12 | wc -l
521
$ ./rt_scrape.py "tn/hamilton-county/chattanooga" 12 > chatt_houses.csv
$ head -3 ex1data.txt
1980,45000
1686,30660
1097,20000

## Setup ##
$ pip install lxml
$ pip install requests-futures
$ chmod +x rt_scrape.py
"""

from sys import argv, stderr
from lxml import html
from requests_futures.sessions import FuturesSession
async_request = FuturesSession(max_workers=20)

def parse_house(house):
    info1 = house.xpath('div[@class="basicdata"]/div[@class="characteristics"]/dl/span[@class="propertyInfo"]')
    info2 = house.xpath('div[@class="price-info"]/dl/span[@class="propertyInfo"]')
    zipcode = house.xpath('span[@itemprop="postalCode"]')
    beds,baths,sqft = (p.xpath ('dd/text()') for p in info1)
    price,date,_ = (p.xpath('dd/text()') for p in info2)
    fields = [sqft,price]
    if any(x == [] or x == [u'\xa0'] for x in fields):
        return None
    return (x[0].strip('\r\n $').replace(',','') for x in fields)

def parse_page(page):
    tree = html.fromstring(page.content)
    houses = tree.xpath('//div[@id="housesList"]/div/div[@class="content"]')
    return filter(lambda x: x != None, (parse_house(h) for h in houses))

def url(query, page):
    return "http://www.realtytrac.com/mapsearch/sold/" + \
            "%s/p-%d?sortbyfield=proximity,asc&itemsper=50" % (query, page)

def get_pages(query,n_pages):
    return [async_request.get(url(query,p)) for p in range(1,n_pages+1)]

def rt_scrape(query,n_pages):
    return (h for p in get_pages(query, n_pages)
              for h in parse_page(p.result()))

def main():
    try:
        query = argv[1]; n_pages = int(argv[2])
        for h in rt_scrape(query,n_pages):
            print ','.join(h)
    except:
        print >> stderr, "Error\n"+__doc__

if __name__ == "__main__":
    main()
