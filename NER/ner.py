import pandas as pd
import spacy
df2 = pd.read_csv('../generatedData/ata2.csv')
nlp = spacy.load("en_core_web_lg")

person=[] #People including fictional.
fac=[] #Buildings, airports, highways, bridges,etc.
org =[] #companies, agencies, institutions,etc.
gpe =[]  #countries,cities,states
loc =[] #non gpe locations, mountaion ranges, bodies of water.
product=[] #objects, vehicles, foods,etc.
event=[]  #Named hurricanes, battles,wars, sports,events,etc.
work_of_art=[] #Titles of books, songs, etc.


for i in range(0,df2.shape[0]):
    temp1=[]
    temp2=[]
    temp3=[]
    temp4=[]
    temp5=[]
    temp6=[]
    temp7=[]
    temp8=[]
    
    a=df2['Description'][i]
    doc = nlp(a)
    for entity in doc.ents:
        
        if entity.label_ =='PERSON':
            temp1.append(entity.text)
        elif entity.label_ =='FAC':
            temp2.append(entity.text)
        elif entity.label_ =='ORG':
            temp3.append(entity.text)
        elif entity.label_ =='GPE':
            temp4.append(entity.text)
        elif entity.label_ =='LOC':
            temp5.append(entity.text)
        elif entity.label_ =='PRODUCT':
            temp6.append(entity.text)
        elif entity.label_ =='EVENT':
            temp7.append(entity.text)
        elif entity.label_ =='WORK_OF_ART':
            temp8.append(entity.text)
    person.append(temp1)
    fac.append(temp2)
    org.append(temp3)
    gpe.append(temp4)
    loc.append(temp5)
    product.append(temp6)
    event.append(temp7)
    work_of_art.append(temp8)

df2['Person'] = person
df2['Facility']=fac
df2['Organisation']=org
df2['Geopolitin']=gpe
df2['Non_Gpe_loc']=loc
df2['Products']=product
df2['Events']=event
df2['Word_Of_Art']=work_of_art
df2.to_csv('../generatedData/LabeledData.csv')
    
    
