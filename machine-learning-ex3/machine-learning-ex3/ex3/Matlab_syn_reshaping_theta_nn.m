%How to turn parameter and gradient matrices into vectors for optimization
%functions and turn then back into matrices.

Theta1=ones(10,11);
Theta2=2*ones(10,11);
Theta3=3*ones(1,11);

Theta1
Theta2
Theta3

ThetaVec=[Theta1(:);Theta2(:);Theta3(:)];
size(ThetaVec)
ThetaVec

reshape(ThetaVec(1:110),10,11)
reshape(ThetaVec(111:220),10,11)
reshape(ThetaVec(221:232),1,11)
