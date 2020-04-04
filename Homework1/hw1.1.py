import numpy as np
import time as t
import matplotlib.pyplot as plt 
#definizione funzione di attivazione: nello specifico Sigmoid
def f(x):
    return 1/(1+np.exp(-x))


X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])


y = np.array([[0,0,1,1]]).T


#inizializzazione matrice pesi
np.random.seed([42])

# initialize weights randomly with mean 0
W = 2*np.random.random((3,1)) - 1
#Stampo la matrice 3 righe 1 colonna dei pesi
#print(W)
#print("\n\n")

num_iters = 1000

learning_rate = 0.5
#creo un vettore mse che contiene all'inizio 1000 elmenti tutti inizializzati a 0.
mse = np.zeros(num_iters)
# For-loop for the iterations
for it in range(num_iters):

    #gira 4 volte perchè quattro sono le righe della matrice
    for n in range (len(X)):
        #prende ogni singola riga del vettore x
        x_n = np.reshape(X[n], (3,1))

        #prende il primo elemento dell array y target.
        #Ovviamente per ogni riga della matrice d'input X è associato un elemento della y targer
        y_target = y[n]

        #print(y_target)

##############Applichiamo la Forward propagation######################################
        y_out = f(np.dot(W.T, x_n))
        print("Guardalo-->: \n",y_out)
        #print(mse[it])
        mse[it] += pow((y_target - y_out),2)
        #print(mse[it])
        #print(mse)
        t.sleep(1000)
 ###########Inizio la parte di Back-propagation############################

        #computi il gradiente

        grad = (y_out -y_target) * y_out * (1 - y_out)
        W_delta = -learning_rate * grad * x_n
        W += W_delta

######Calcolo MSE###################################
    mse[it] /= len(X)

#Calcolo y stimato finale dopo aver completato i 1000 cicli
y_out = f(np.dot(X, W))
print("Output after training, y_out")
print(y_out)
print("Target output, y")
print(y)



############################################
#Semplice visualizzazione dei vari grafici
x1 = np.arange(-4,4,.01)
plt.figure()
plt.plot(x1, np.maximum(x1,0), label='ReLu')
plt.plot(x1, 1/(1+np.exp(-x1)), label='sigmoid')
plt.plot(x1, np.tanh(x1), label='tanh')
plt.axis([-4, 4, -1.1, 1.1])
plt.title('Activation functions')
l = plt.legend()
plt.savefig('C:\\Users\\claud\\Desktop\\grafico1.png')
################################################
plt.figure()
plt.plot(range(num_iters), mse, label="MSE")
l = plt.legend()
plt.savefig('C:\\Users\\claud\\Desktop\\grafico2_finale.png')