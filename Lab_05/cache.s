.data
array:	.word	2048

.text

# change the following parameters: ####
	li		a0,	512		# size of array
	li		a1,	16		# step size
#######################################
	la		t1, array
	add		t3,t1,a0
array_loop:
	lb		t2,0(t1)
	add		t1,t1,a1
	blt 	t1,t3,array_loop
exit:    
	li	a0,10
	ecall
