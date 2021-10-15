# some comments

- splitting out the data pre-processing from the Adaboost with Smote code while a good idea in principle makes it harder to identify what we need to do to modify the code to allow X to update as Adaboost runs
- I think we need to see the other older version to be able to see how to modify it, ebcause of the way you are now structuring the data pre-processing. It was clearer before.
- My thoughts from last week were to make a copy of X then as you update the weights in parallel with that update the copy of X as well.
- found it. have added some comments in the 2 class file as to where I think smote needs to be called.

# Today's discussion

Carry on with literature start writing in Overleaf. Use GIang's datasets

- Experiment 1

Make a function that makes it class balanced using SMOTE line 43 initialize data make it class-balnced by SMOTE

- Experiment 2

call function line 58

- Experiment 3

--  3.1 Experiment 1 and 2 change line 58 to a probability user defined constant SMOTE rate
Call sMOTE earlier or later?

-- 3.2 decrease SMOTE rate over time

 
