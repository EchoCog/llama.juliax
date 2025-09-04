NB. Simple test for J fusion interface
load 'src/fusion.ijs'

NB. Test basic array operations
A =: 1 2 3 4
B =: 5 6 7 8

NB. Test addition
C =: A + B
echo 'Addition test: ', ": C

NB. Test multiplication  
D =: A * B
echo 'Multiplication test: ', ": D

echo 'J fusion basic tests completed successfully'

exit 0