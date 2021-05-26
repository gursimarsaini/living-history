import pandas as pd

df = pd.read_csv('generatedData/LabeledData.csv')


def Query(ab,category):
    hd={}
    temp_indexes=[]
    typoe=[]
    Number_of_types=0
    ax=True
    for i in range(0,df.shape[0]):
        #print(df['Organisation'][i])
        li = list(df[category][i].split(", ")) 
    
        for j in li:
            f=j.replace("'","")
            g= f.replace("[","")
            h =g.replace("]","")
        
            #print('hi',h )
        
            if h.lower()==ab:
                ax=False
                #print(j,"  ==  ",ab)
                #hd.append(i)
                temp_indexes.append(i)
                #saving new index and its catergory in hd dictionary
                hd[i]=df['Type'][i]
                
                #insertine unique types in typoe list
                if hd[i] not in typoe:
                    typoe.append(df['Type'][i])
                    #print('hi')
                    Number_of_types = Number_of_types+1
                    
    if ax:
        return 0,0

    #print(typoe)
    #calling for each type
    
    #indexs of each type under that key in dictionary, eg. all indexes of type india under dict[india]
    type_index_dict={}
    #print(typoe)
    #print(temp_indexes)
    
    for i in typoe:
        temp=[]
        for j in hd:
            if hd[j]==i:
                temp.append(j)
        type_index_dict[i]=temp
    #print(type_index_dict)
    
    for i in type_index_dict:
        temp={}
        for j in type_index_dict[i]:
            #print(j)
            if j in temp:
                temp[j]=temp[j]+1
            else:
                temp[j]=1
        type_index_dict[i]=temp
        
    for i in type_index_dict:
        temp={}
        for j in type_index_dict[i]:
            temp[j]=temp_indexes.count(j)
        type_index_dict[i]=temp
            
        
#     dict={}
#     for i in temp_indexes:
#         if i in dict:   
#             dict[i]=dict[i]+1
#         else:
#             dict[i]=1
    #print(type_index_dict)
    for i in type_index_dict:
        d=html(type_index_dict[i])
        type_index_dict[i]=d
    
    #d=html(dict)
    
    return type_index_dict,Number_of_types

def html(dictionary):
    indexes = sorted(dictionary,key=dictionary.get,reverse=True)[:5]
    
    html_date_list={}
    if len(indexes)>=5:
        for i in range(0,len(indexes)):
            vd={}
            Dater = df['Date'][indexes[i]]
            vd['date'] =Dater
            headliner=df['Headlines'][indexes[i]]
            vd['headlines']= headliner
            descriptionr = df['Description'][indexes[i]]
            vd['description'] = descriptionr.replace("\n","")
            #a = displacy.render(nlp(str(df['Description'][indexes[i]])), jupyter=False, style='ent')
            #vd['tags'] = a.replace("\n","")
            #print(indexes[i])
            
            doc = nlp(descriptionr.replace("\n",""))
            li=[]
            for entity in doc.ents:
                li.append(entity.text)
            vd['tags']=li
            #vd['tags'] = a.replace("\n","")
            #print(indexes[i])
            #html_date_list[i] = vd
            
            html_date_list[i]=vd
    else:
        for i in range(0,len(indexes)):
            vd={}
            Dater = df['Date'][indexes[i]]
            vd['Date']=Dater
            headliner=df['Headlines'][indexes[i]]
            vd['headlines'] = headliner
            descriptionr = df['Description'][indexes[i]]
            vd['description'] =descriptionr.replace("\n","")
            #a = displacy.render(nlp(str(df['Description'][indexes[i]])), jupyter=False, style='ent')
            doc = nlp(descriptionr.replace("\n",""))
            li=[]
            for entity in doc.ents:
                li.append(entity.text)
            vd['tags']=li
            #vd['tags'] = a.replace("\n","")
            #print(indexes[i])
            html_date_list[i] = vd
            
    #print(html_date_list)
    return html_date_list


query="sc"
category='Organisation'
e,c=Query(ab,category)
    


