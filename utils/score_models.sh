#!/bin/bash

# This would have been my preferred BLEU score metric however, the MOSES method for BLEU is now failing to compile on clara, jenny, and river

# echo "SCORING BLEU FOR NON-EMBED DEV"
# $MOSES/scripts/generic/multi-bleu.perl -lc /workspace/data/dev.nl < /workspace/data/non_embed_dev.nl
# sleep 0.3
# echo "FINISHED SCORING BLEU FOR NON-EMBED DEV"
#
# echo "SCORING BLEU FOR NON-EMBED TEST"
# $MOSES/scripts/generic/multi-bleu.perl -lc /workspace/data/test.nl < /workspace/data/non_embed_test.nl
# sleep 0.3
# echo "FINISHED SCORING BLEU FOR NON-EMBED TEST"
#
# echo "SCORING BLEU FOR EMBED DEV"
# $MOSES/scripts/generic/multi-bleu.perl -lc /workspace/data/dev.nl < /workspace/data/embed_dev.nl
# sleep 0.3
# echo "FINISHED SCORING BLEU FOR EMBED DEV"
#
# echo "SCORING BLEU FOR EMBED TEST"
# $MOSES/scripts/generic/multi-bleu.perl -lc /workspace/data/test.nl < /workspace/data/embed_test.nl
# sleep 0.3
# echo "FINISHED SCORING BLEU FOR EMBED TEST"

# Generating the translations (comments) for dev and test for both models
python generate.py

# USING NLTK implementation of BLEU score (I think this is different from the MOSES implementation that I used for my baseline model)
python bleu.py data/dev.nl data/non_embed_dev.nl 2> errors.txt
python bleu.py data/test.nl data/non_embed_test.nl 2> errors.txt
python bleu.py data/dev.nl data/embed_dev.nl 2> errors.txt
python bleu.py data/test.nl data/embed_test.nl 2> errors.txt
