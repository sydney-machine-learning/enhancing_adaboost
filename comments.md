# some comments

- splitting out the data pre-processing from the Adaboost with Smote code while a good idea in principle makes it harder to identify what we need to do to modify the code to allow X to update as Adaboost runs
- I think we need to see the other older version to be able to see how to modify it, ebcause of the way you are now structuring the data pre-processing. It was clearer before.
- My thoughts from last week were to make a copy of X then as you update the weights in parallel with that update the copy of X as well.
- with the ne structure of with the data pre-processing in a separate file I'm finding it hard to see how to do that, because X isn't called anywhere in the adaboost file.
