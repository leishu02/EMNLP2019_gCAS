#!/bin/sh
python model.py -domain movie -network gcas -mode test
python model.py -domain taxi -network gcas -mode test
python model.py -domain restaurant -network gcas -mode test
