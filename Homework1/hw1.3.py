import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import load_digits 
import time as t

np.random.seed([42])

digits = load_digits()


sample_index = 45
#plt.figure(figsize=(3, 3))
#plt.imshow(digits.images[sample_index], cmap=plt.cm.gray_r,
#interpolation='nearest')
#plt.title("image label: %d" % digits.target[sample_index])
#plt.savefig('C:\\Users\\claud\\Desktop\\grafico1.png')


#input
data = np.asarray(digits.data, dtype='float32')
#target finale
target = np.asarray(digits.target, dtype='int32')

print("Input X:  %r \n"  %data) #la x ha 1797 di sotto-elementi
print("Lunghezza Input X:  %r \n"  %len(data))

print("Target y: %r \n"%target)#la y è un array con 1797 elementi
print("Lunghezza target: %r \n" %len(target))

def one_hot(n_classes, y):
    return np.eye(n_classes)[y]


###############FUNZIONE DI ATTIVAZIONE SOTFMAX#########################
def softmax(X):
    #exps = np.exp(X)
    return  np.dot((1 / np.sum(np.exp(X),axis=0)),np.exp(X))

#############FUNZIONE DI COSTO##########################################
EPSILON = 1e-8

def cross_entropy(Y_true, Y_pred):

    Y_true, Y_pred = np.atleast_2d(Y_true), np.atleast_2d(Y_pred)
    loglikelihoods = np.sum(np.log(EPSILON + Y_pred) * Y_true, axis=1)
    return -np.mean(loglikelihoods)
##########################################################################
input_size=data.shape[1]
n_classes = len(np.unique(target))

#inizializzazione pesi
W = np.random.uniform(size=(input_size,n_classes),high=0.1, low=-0.1)
print("Inizializzazione vettore pesi: %r \n" %W)
print("Lunghezza inizializzazione vettore pesi: %r \n" %len(W))
#bias
b = np.random.uniform(size=n_classes, high=0.1, low=-0.1)
print("Bias: %r \n" %b)
print("Lunghezza Bias: %r \n" %len(b))

###Consideriamo un campione dal set di addestramento, e tracciamo l'output corrente del nostro modello, prima di addestrarlo.####

'''
y_out=softmax(np.dot(data[sample_index], W) + b)

plt.bar(range(n_classes),y_out,label='prediction', color="red")
plt.ylim(0,1,0.1)
plt.xticks(range(n_classes))
plt.legend()
plt.ylabel("probability")
plt.title("target:"+str(target[sample_index]))
plt.savefig('C:\\Users\\claud\\Desktop\\grafico1.png')
'''
num_iters = 50

learning_rate = 0.0005

for it in range(num_iters):
    iteration_accuracy=[]
    iteration_loss=[]

    #la i è il contatore mentre enumerate(zip(data, target)) ti stampa 
    for i, (X, y) in enumerate(zip(data, target)):

        y_out=softmax(np.dot(X, W) + b)

        pred_err = y_out - one_hot(n_classes, y) 

        #aggiorno i pesi 
        W_delta = np.outer(pred_err,X).transpose()
        W += (W_delta * (-learning_rate))
        #aggiorno il bias
        b += (pred_err * (-learning_rate))



        iteration_accuracy.append(np.argmax(y_out) == y)
        iteration_loss.append(cross_entropy(one_hot(n_classes,y),y_out))

    print("iteration: ",it," -- accuracy: ",np.mean(np.asarray(iteration_accuracy)), " -- loss: ", np.mean(iteration_loss))

y_pred=softmax(np.dot(data[sample_index], W) + b)
plt.bar(range(n_classes), y_pred, label='prediction', color="red")
plt.ylim(0, 1, 0.1)
plt.xticks(range(n_classes))
plt.legend()
plt.ylabel("probability")
plt.title("target:"+str(target[sample_index]))
plt.show()