from sys import argv 
script, filename=argv 

import indicoio
indicoio.config.api_key = '5937b9028ff33bc6b342a1242ee39dad'

import numpy
import matplotlib.pyplot as plt

#read the file line by line
with open(filename) as f:
    lines = f.readlines()


#send request to Indico SA API and store the scores in a list
sent_scores=[]
for l in lines:
	  sent_scores.append(indicoio.sentiment_hq(l))
	 
#print ("All scores %s:" %sent_scores)


#map score ranges to categories(arbitrarily assigned here)
pos_total=0
neutral_total=0
negative_total=0

for l in lines:
	  if indicoio.sentiment_hq(l)<0.3:
	      negative_total+=1
	  elif indicoio.sentiment_hq(l)>=0.3 and indicoio.sentiment_hq(l)<0.7:
	      neutral_total+=1
	  else:
	      pos_total+=1
	  
print("Total positive sentences: %s." %pos_total)
print("Total neutral sentences: %s." %neutral_total)
print("Total negative sentences: %s." %negative_total)


#create values for y using the stored scores
y_value=numpy.array(sent_scores)
n=len(y_value)

#create values for x using the number of lines
x_value=numpy.arange(1,(n+1), 1)

#create values for the mean line using the mean score multiplied by the number of lines
mean_number=numpy.mean(y_value)
print ("Mean Sentiment Score: %r" % mean_number)
y_mean_value=numpy.array(len(x_value)*[mean_number])

fig=plt.figure()

#subplot1
ax1=fig.add_subplot(211)

data_line=ax1.plot(x_value,y_value,'b',label='Sentiment Score', marker='o')
mean_line=plt.plot(x_value,y_mean_value,'#660000',label='Mean Score',linestyle='--')

# add axis labels
plt.xlabel('Text Flow')
plt.ylabel('Sentiment Score')

#add legend
legend = plt.legend(loc='best', shadow=True)
legend.get_frame().set_facecolor('#ccffff')

#add title
plt.title('Sentiment Score Flow')

#subplot2
ax2=fig.add_subplot(212)
ax2.hist(y_value,bins = [0.0,0.2,0.4,0.6,0.8,1.0],histtype='bar')

#add axis labels
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")

#add grid
plt.grid(True)

#add title
plt.title("Sentiment Score Frequency")

plt.subplots_adjust(hspace=0.45)
plt.show()