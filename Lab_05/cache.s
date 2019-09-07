.data
array:	.word	2048

.text

## change the following parameters: ########################
	li		a0,16		# number of accesses
	li		a1,64		# step size
	li		a2,4		# rep count
	li		a3,1		# offset
############################################################

	li		t3,0
rep_loop:
	addi	t4,a0,0
	la		t1,array
	add		t1,t1,t3
array_loop:
	lb		t2,0(t1)
	add		t1,t1,a1
	addi	t4,t4,-1
	bne 	t4,x0,array_loop
	add		t3,t3,a3
	bne		t3,a2,rep_loop

exit:
	li		a0,10
	ecall

