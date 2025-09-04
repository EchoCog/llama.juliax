NB. Test file for J fusion interface
NB. This script tests basic functionality of the J fusion bindings

NB. Load the fusion interface
load 'bindings/j/src/fusion.ijs'

NB. Test 1: Basic array operations
echo 'Testing basic array operations...'
A =: 2 3 $ 1 2 3 4 5 6
B =: 3 2 $ 7 8 9 10 11 12

NB. Test addition
C =: A add A
expected_C =: 2 3 $ 2 4 6 8 10 12
assert C -: expected_C

echo 'Array addition: PASS'

NB. Test 2: Mathematical operations
echo 'Testing mathematical operations...'

NB. Test sum operation
sum_A =: SIGMA A
expected_sum =: 21
assert sum_A = expected_sum

echo 'Mathematical sum: PASS'

NB. Test 3: Help system
echo 'Testing help system...'
help 'beginner'
help 'intermediate'  
help 'expert'

echo 'Help system: PASS'

NB. Test 4: Status check
echo 'Testing status...'
status''

echo 'Status check: PASS'

NB. Test 5: Version information
echo 'Testing version...'
version''

echo 'Version check: PASS'

NB. Summary
echo ''
echo 'All J fusion interface tests passed!'
echo 'J interface is working correctly.'

exit 0