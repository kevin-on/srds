adaptive = 4

for srds_iter in range(10):
	if srds_iter < adaptive:
		iter_scheduler_fine = 0
	else:
		iter_scheduler_fine = 1

	for i in range(1, 10 + 1):  # line 7 of Algorithm 1
		# Scheduling pattern:
		# srds_iter=0: fine, fine_sub, fine_sub, ...
		# srds_iter=1: fine, fine, fine_sub, fine_sub, ...
		# srds_iter=2: fine, fine, fine, fine_sub, fine_sub, ...
		if i <= srds_iter + 1:
			cur_scheduler_fine = 1
		else:
			cur_scheduler_fine = iter_scheduler_fine
		print(cur_scheduler_fine, end=" ")
	print("\n")