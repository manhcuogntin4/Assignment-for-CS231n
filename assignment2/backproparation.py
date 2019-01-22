#!/bin/python3
import   math
import   os
import   random
import   re
import   sys
import   ast
import   numpy as  np
class OpConv2D:
	def  __init__(self, filters_count, kernel_size,inputs):
		self.inputs = inputs
		self.kernel_size = kernel_size
		input_height =  inputs.shape[0]
		input_width  =  inputs.shape[1]
		input_channels =  inputs.shape[2]
		self.output_height =  input_height - kernel_size +  1
		self.output_width =  input_width - kernel_size +  1
		self.output_channels =  filters_count
		self.weights =  np.random.normal(size=(filters_count,kernel_size,kernel_size,input_channels),scale=0.1)

	def foreward(self):
		Z =  np.zeros((self.output_height,self.output_width,self.output_channels),dtype=np.float32)
		for h in range(self.output_height):
			for w in range(self.output_width):
				for c  in  range(self.output_channels):
					z  =  0
					for i in range(self.kernel_size):
						for j in range(self.kernel_size):
							for k in range(self.weights.shape[-1]):
								z  +=  self.inputs[h+i,w +j,k]*self.weights[c][i][j][k]
								Z[h,w,c]  =   z
		return Z

	def backward(self,dZ):
		dW = np.zeros(self.weights.shape, dtype=np.float32)
		dA_prev  =  np.zeros(self.inputs.shape,dtype=np.float32)
		(n_H_prev, n_W_prev,n_C_prev) = dA_prev.shape
		(fc,f, f,c)=dW.shape
		(n_H_Z,n_W_Z,n_C_Z)=dZ.shape
		# Initializing dX, dW with the correct shapes
		dX = np.zeros(dA_prev.shape)
		dW = np.zeros(dW.shape)
		 # Looping over vertical(h) and horizontal(w) axis of the output
		H,W,D=self.inputs.shape
		for t in range(fc):
			for h in range(n_H_Z):
				for w in range(n_W_Z):
					for d in range(D):
						dA_prev[h:h+f,w:w+f,d]+= self.weights[t,:,:,d]* dZ[h,w,t]
						dW[t,:,:,d]+= self.inputs[h:h+f, w:w+f,d] * dZ[h,w,t]

		return (dW,dA_prev)

def sround(item):
	if type(item) is   list:
		return [sround(it) for it  in item]
	else:
		return round(item,3)

if __name__ == '__main__':
	#pA=ast.literal_eval(sys.stdin.readline())
	#x=sys.stdin.read()
	#x=ast.literal_eval(x)
	#x=[[[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8]]]
	#x=np.array(x,np.float32)
	#x=np.array([[[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.3,0.7,0.8]]],np.float32)
	x=np.array([[[0.05658516744757707],[1.044554184159321], [0.015219418258455185]], [[0.11938270577254026], [0.30541981019155275], [0.7672828761835571]], [[0.2503537591720918], [0.04351723514830752], [0.5212979987199301]]])
	#x=np.random.randn(192)
	print(x.shape)
	#x=x.reshape((8,8,3))
	x=np.array(x,np.float32)
	#print(x.shape)
	#np.random.seed(1)
	op=OpConv2D(4,3,x)
	Z=op.foreward()
	print Z
	dZ=np.ones(Z.shape,dtype=np.float32)
	#print dZ
	dW,dA=op.backward(dZ)
	print(sround(dW.tolist()))



