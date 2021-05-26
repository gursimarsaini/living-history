
# coding: utf-8

# In[1]:


import pandas as pd
import os
import shutil
import time
# In[4]:


import networkx as nx
import matplotlib.pyplot as plt


# In[31]:


df=pd.read_csv('generatedData/relations.csv')

# In[32]:


#df.head()


# In[53]:


def filter_graph(pairs, node):
    # from_pandas_edgelist: Returns a graph from Pandas DataFrame containing an edge list.
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
            create_using=nx.MultiDiGraph())
            # MultiDiGraph: directed graph class that can store multiedges
    edges = nx.dfs_successors(k_graph, node) #Returns dictionary of successors in depth-first-search from source.
    nodes = []
    for k, v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    subgraph = k_graph.subgraph(nodes)
    layout = (nx.random_layout(k_graph)) #Position nodes uniformly at random in the unit square.
    #attributes for the graph
    nx.draw_networkx(
        subgraph,
        node_size=300,
        arrowsize=8,
        linewidths=4,
        pos=layout,
        edge_color='grey',
        font_size=6,
        font_color='grey',
        node_shape="s",
        #edgecolors='black',
        node_color='skyblue'
        )
    labels = dict(zip((list(zip(pairs.subject, pairs.object))),
                    pairs['relation'].tolist()))
    edges= tuple(subgraph.out_edges(data=False))
    sublabels ={k: labels[k] for k in edges}
                                    #edge attributes
    nx.draw_networkx_edge_labels(subgraph, pos=layout, edge_labels=sublabels,
                                font_color='red')
    plt.axis('off')
    #plt.show()
    newGraph = "result" + str(time.time()) + ".png"
    for filename in os.listdir('static/graph/'):
        if filename.startswith('result'):
            os.remove('static/graph/' + filename)
    plt.savefig('static/graph/'+newGraph)
        #saving as a image file
    rslt='graph/'+newGraph
    return rslt


# In[58]:


def Query_graph(query):
    try:
        rslt=filter_graph(df,query)
        return rslt
    except:
        for filename in os.listdir('static/graph/'):
            if filename.startswith('result'):
                os.remove('static/graph/' + filename)
        src='static/images/no-result.png'
        noGraph = "result" + str(time.time()) + ".png"
        dst='static/graph/'+noGraph
        shutil.copy(src,dst)
        rslt='graph/'+noGraph
        return rslt


# In[60]:


#Query_graph("road")

#Query_graph("women")
