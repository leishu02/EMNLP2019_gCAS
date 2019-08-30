#!/bin/sh
python model.py -domain movie -network gcas -mode train -cfg cuda=True
python model.py -domain taxi -network gcas -mode train -cfg cuda=True
python model.py -domain restaurant -network gcas -mode train -cfg cuda=True

