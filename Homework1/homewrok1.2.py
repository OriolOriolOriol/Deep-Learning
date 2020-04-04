import numpy as np 
import time as t 
import matplotlib.pyplot as plt

def f(x):
    return 1/(1 + np.exp(-x))

def f_derivato(x):

    return x * (1 -x)

#Creo il mio input
X= np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
print("Input: \n", X)
#Ecco il mio target (Vettore colonna)
y = np.array ([0,1,1,0]).T
print("Target: \n", y)
#genero numeri pseudo-casuali
np.random.seed([42])


#numero di neuroni nello strato hidden layer
num_hidden = 4

#inizializzo i pesi in modo random con media 0
#W_1 = hidden_weights
#W_2 = output weights

W_1= 2*np.random.random((3,num_hidden)) -1 # i pesi da input -> hidden
print("\nPesi hidden layer: \n", W_1)

W_2 = 2*np.random.random((num_hidden,1)) -1 # i pesi da hidden -> output
print("Pesi output layer: \n", W_2)


num_iters = 5000 #epoche
eps= 0.5 #learning rate

mse = np.zeros(num_iters)


for it in range (num_iters):

    for n in range (len(X)): #cicla 4 volte perchè ci sono 4 elementi

        x_n = np.reshape(X[n],(3,1)) #per ogni elemnto di X fai prendi i 3 elementi li suddivisi in modo tale da creare 3 cerchietti di input
       # print("I primi 3 nodi di X: %r \n" %x_n)

        y_target= y[n]

        #print("y-Target: %r \n" %y_target)
        
        ############FORWARD PROPAGATION#################################################
        h = f(np.dot(W_1.T, x_n))
        y_out = f(np.dot(W_2.T, h))

        ###########BACK PROPAGATION#####################################################
       # print("Result on hidden layer: %r \n" %h)
        #print("Predict: %r \n"%y_out)

       # print("\n\nLIVELLO TRA OUTPUT E HIDDEN LAYER\n\n")
        delta_output= y_out * (1- y_out) * (y_target - y_out)
        #print("Delta output: %r \n" %delta_output) #rappresenta la derivata dello strato nascosto (dell'unità di y)

        delta_hidden= np.zeros(len(W_2))
        #print("Delta hidden Iniziale: %r \n" %delta_hidden)#il delta delle frecce rosse nel disegno

        #print("Delta Hidden ciclato \n")
        for i in range(len(W_2)):
            delta_hidden[i] = delta_output * h[i]
            print(delta_hidden)

        #print("\n")

        #print("\n\nLIVELLO TRA HIDDEN LAYER E INPUT\n\n")
        delta_hidden_input =np.zeros(4)
        #print("Delta hidden input Iniziale: %r \n" %delta_hidden_input)
        delta_input =np.zeros([3,4])
        #print("Delta Input Iniziale: %r \n" %delta_input)


        print("Delta hidden input:  \n")
        for j in range (num_hidden):
            delta_hidden_input[j]= h[j,0] * (1 - h[j,0]) * (W_2[j] * delta_output)

            for i in range (len(W_1)):
                delta_input[i,j] = delta_hidden_input[j] * x_n[i,]
            #print("\n")   


    #aggiorno dei pesi
        for i in range (len(W_2)):
            W_2[i] += eps * delta_hidden[i]

        for (i,j),_ in np.ndenumerate (W_1):

            W_1 [i,j] += eps * delta_input[i,j]
        

y_out = f(np.dot(f(np.dot(X, W_1)), W_2))
print("Output after training, y_out")
print(y_out)
print("Target output, y")
print(y)