#!/bin/sh
python model.py -domain movie -network gcas -mode eval
python model.py -domain taxi -network gcas -mode eval
python model.py -domain restaurant -network gcas -mode eval
