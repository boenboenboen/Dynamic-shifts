import itertools
import pickle
import os
import ast
from collections import Counter
from tqdm import tqdm
import networkx as nx
import json
from networkx.algorithms import community
import argparse
import csv
def get_university(org):
    
    st = org.split(",")
    
    st = [item.strip() for item in st]
    
    for x in st:
        if 'niversit' in x:
            return x


labels = {}

USED_ORG = ['Anthropic', 'Microsoft Research Lab – AI',
            'Allen Institute for AI', 'Facebook AI Research (FAIR)', 'DeepMind Lab',
            'NVIDIA Research',
            'Vector Institute', 'Intel AI lab',
            'Google',
            'IBM Research', 'Amazon Web Services AI Lab', 'OpenAI']


parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, default=0, help='Specify the stage number (default: 0)')
args = parser.parse_args()

period = [(0,4), (4,7), (7,10)]

edges = list()

avg_cite = {}
for num in range(period[args.stage][0],period[args.stage][1]):
    print(num)
    file_name = '10_graph_update/' + str(num) + '.json'
    with open(file_name,'r') as json_file:
        data = json.load(json_file)
    
    #edges = list()
    
    for paper in data:

        orgs = paper['org_set']
        orgs = list(set(orgs))
        if len(orgs) <= 1:
            continue
        

        for org in orgs:
            if org not in avg_cite:
                avg_cite[org]= []
            
            try:
                avg_cite[org].append(int(paper['cited_num']))
            except:
                avg_cite[org].append(0)
        for org1, org2 in itertools.combinations(orgs, 2):
            
            if 'niversit' in org1:
                org1 = get_university(org1)
                labels[org1] = 'University'
            else:
                if org1 in USED_ORG:
                    labels[org1] = 'Key_org'
                else:
                    labels[org1] = 'Org'
            
            if 'niversit' in org2:
                org2 = get_university(org2)
                labels[org2] = 'University'
            else:
                if org2 in USED_ORG:
                    labels[org2] = 'Key_org'
                else:
                    labels[org2] = 'Org'
            
            
            edges.append((org1, org2))


avg_ci = {}
for x in avg_cite:
    if len(avg_cite[x])>0:
        avg_ci[x] = sum(avg_cite[x]) / len(avg_cite[x])

'''
for x in USED_ORG:
    if x in avg_ci:
        
        print(x, len(avg_cite[x]), avg_ci[x])
'''

USED_ORG = ['Anthropic', 'Microsoft Research Lab – AI',
            'Allen Institute for AI', 'Facebook AI Research (FAIR)', 'DeepMind Lab',
            'NVIDIA Research',
            'Vector Institute', 'Intel AI lab',
            'Google',
            'IBM Research', 'Amazon Web Services AI Lab', 'OpenAI']

lab = {'Anthropic':'u', 'Microsoft Research Lab – AI':'w', 'Allen Institute for AI':'u','Facebook AI Research (FAIR)':'w', 
       'DeepMind Lab':'u', 'NVIDIA Research':'w', 'Vector Institute':'u',  'Intel AI lab':'w', 'Google':'w', 'IBM Research':'w',  'Amazon Web Services AI Lab':'w', 'OpenAI':'u'}
relationship_count = Counter(tuple(sorted(pair)) for pair in edges)
edgelist = [(pair[0], pair[1]) for pair, count in relationship_count.items()]
    
    
weights = [ count for pair, count in relationship_count.items()]
G = nx.Graph()
total = {}
to_whale  = {}
to_unicorn = {}
to_university = {}
for x in USED_ORG:
    total[x] = 0
    to_whale[x] = 0
    to_unicorn[x] = 0
    to_university[x] = 0
for i in range(0,len(edgelist)):
    if edgelist[i][0] != edgelist[i][1]:    
        G.add_edge(edgelist[i][0],edgelist[i][1], weight = weights[i])
        x, y= edgelist[i][0], edgelist[i][1]
        if x in USED_ORG:
            total[x] += weights[i]
            if y in USED_ORG:
                if lab[y] == 'w':
                    to_whale[x] += weights[i]
                else:
                    to_unicorn[x] += weights[i]
                
            elif 'niversit' in y:
                to_university[x] += weights[i]

        x,y = y,x
        if x in USED_ORG:
            total[x] += weights[i]
            if y in USED_ORG:
                if lab[y] == 'w':
                    to_whale[x] += weights[i]
                else:
                    to_unicorn[x] += weights[i]
                
            elif 'niversit' in y:
                to_university[x] += weights[i]



for x in USED_ORG:
    if total[x] ==0:
        continue
    to_whale[x] /= total[x]
    to_unicorn[x] /= total[x]
    to_university[x] /= total[x]


order = ['Google','Microsoft Research Lab – AI',
            'IBM Research', 'NVIDIA Research',  'Intel AI lab','Amazon Web Services AI Lab', 'Facebook AI Research (FAIR)','Allen Institute for AI', 'Anthropic', 'OpenAI','DeepMind Lab','Vector Institute']


#nx.set_node_attributes(G,labels, 'label')


effective_size = nx.effective_size(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
#efficiency = nx.global_efficiency(G)


with open(f'qca-{args.stage}.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['Institute', 'Betweenness Centrality', 'Closeness Centrality', 'Degree Centrality', 'Effective Size', 'Efficiency', 'Average Citation', 'To-Large', 'To-Frontier', 'To-University'])
    for institute in order:
        try:
            ego_net = nx.ego_graph(G, institute)
            num_alters = len(ego_net.nodes) - 1
            writer.writerow([institute, betweenness_centrality[institute], closeness_centrality[institute], degree_centrality[institute], effective_size[institute], effective_size[institute]/num_alters,  avg_ci[institute], 
                         to_whale[institute], to_unicorn[institute], to_university[institute]])
        except:
            writer.writerow([institute, 0, 0, 0, 0, 0, 0, 0, 0, 0])

'''    
print(len(G.nodes))
output_file = str(num) + '_updated.gexf'
nx.write_gexf(G, output_file)
'''


'''
eid
paper_id
source_id
source_title
paper_title
paper_authors_name
paper_authors_id
pub_year
cited_num
paper_type
first_author_id
author_keyword_json
index_keyword_json
group_subject_id
group_asjc
doi
language
authors_info
affiliations_info
source_info
open_access
author_keywords
collaborations
funding_list
correspondences
indexed_keywords
open_access_types
publication_stage
view_at_repository_link
org_set
in_domain_org_set
has_whale
has_unicorn
'''