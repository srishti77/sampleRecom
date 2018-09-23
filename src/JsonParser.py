
import json
from pprint import pprint

with open('D:\\Recommender System\\Raw Data\\modified_cloud2.txt') as f:
    data = json.load(f)

pprint(data)