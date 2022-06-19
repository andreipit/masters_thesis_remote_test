#######################################
Task 11 june 2022:
#######################################
Why processor.model.nonlocal_vars[executing_action]=True
doesn't move robot?




#######################################
Task 10 june 2022:
#######################################
context: 
	we can run full loop 
	just processor loop is not parallell
task:
	pick methods 1 by 1 from main loop
	create test file
	reconstruct same state from start to i-th method
	test this method in loop, using input
subtask:
	wrap entry_point lines in 5 big methods
	will help to reproduce starting state for tests




#######################################
# Some notes:
#######################################
#Simulator().test()
# 0) you must launch coppelia scene before all
# 1) convert cmd string to variables


# Assumptions:
# - all comments are given for 1.2 case

# script arguments:
# 1.1) --stage grasp_only --num_obj 5 --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 4 --experience_replay
# 1.2) --stage grasp_only --num_obj 5 --goal_conditioned --goal_obj_idx 4 --experience_replay --explore_rate_decay --save_visualizations
# 1.3) --stage grasp_only --num_obj 5 --grasp_goal_conditioned --goal_conditioned --goal_obj_idx 4 --experience_replay --explore_rate_decay --save_visualizations --grasp_explore --load_explore_snapshot --explore_snapshot_file logs\\agnostic_grasp\\models\\snapshot-backup.grasp_only.pth


# Tools - options - python - analysis - set (workspace, strict)

# Links:
# How to Make Python Statically Typed:
# https://towardsdatascience.com/how-to-make-python-statically-typed-the-essential-guide-e087cf4fa400
# https://numpy.org/devdocs/reference/typing.html
# https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# https://github.com/ramonhagenaars/nptyping
# https://linuxtut.com/en/1f7416bbb29f48a153bb/

# pip install nptyping==2.0.1