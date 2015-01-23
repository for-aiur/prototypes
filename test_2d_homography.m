function test_2d_homography()
clear;
clc;

pointsL = [
    1552        693         1;
    1407.7      1767.7      1;
    2478.4      317.8       1;
    2396.2      1447.1      1;
    1710.3      1324.3      1;
    1828.7      1696.0      1;
    1363.3      1810.7      1;
    1434.0      1080.7      1;
    1724.7      90.0        1;
    1952.7      39.3        1;
    2100.7      811.3       1;
];

pointsR = [
    249         726         1;
    181.3       1879.3      1;
    1110.4      384.7       1;
    1145.5      1425.5      1;
    492.7       1372.3      1;
    649.5       1734.0      1;
    130.0       1933.0      1;
    146.5       1140.0      1;
    415.5       121.0       1;
    700.0       93.0        1;
    848.0       845.0       1;
];

H = calc_2d_homography(pointsL(1:4,:), pointsR(1:4,:))

% Minimize the function using Levenberg-Marquardt
Ho = lsqnonlin(@(h)symetricTransferError(pointsL',pointsR',h),H)

imgL = imread('left.jpg');
imgR = imread('right.jpg');
imgBig = [imgL zeros(size(imgL))];

iH = inv(H);

for x=size(imgL,2):size(imgBig,2)*0.6
    for y=600:size(imgBig,1)*0.6
        p = [x y 1];
        pnew = iH*p';
        pnew = pnew / pnew(3);
        pnew = int32(pnew);
        imgBig(y,x,1:2) = imgR(pnew(2),pnew(1),1:2);
    end
end

imshow(imgBig);
end

function [e] = symetricTransferError(X1,X2,H) 
    % Transform the points
    X2p = H^(-1)*X1;
    X2p = X2p/X2p(3,:);
    X1p = H*X2;
    X1p = X1p/X1p(3,:);

    % First compute the error
    e = sqrt((X1(1,:)'-X1p(1,:)').^2 + (X2(2,:)'-X2p(2,:)').^2);
end

%calculates 2d homography
function [ H ] = calc_2d_homography(pointsL, pointsR)
n = size(pointsL,1);
A = [];

T       = NormalizePoints(pointsR);
T_dash  = NormalizePoints(pointsL);

tempR = T*pointsR';
tempL = T_dash*pointsL';
pointsR_n = tempR';
pointsL_n = tempL';

%fill the matrix 'A'
for i=1:n
    lPoint = pointsL_n(i,:);
    rPoint = pointsR_n(i,:);
    new_segment = [
        zeros(1,3)         -lPoint(3)*rPoint       lPoint(2)*rPoint;
        lPoint(3)*rPoint    zeros(1,3)            -lPoint(1)*rPoint;
        ];
    A = [A;new_segment];
end
[S V D] = svd(A);
%find min index from V instead of using 'end'
h = D(:,end);

disp(sprintf('Mean algebric Error: %d', abs(mean(A*h)) ));

H = [h(1:3)';h(4:6)';h(7:9)'];

%reverse conditioning
H = inv(T_dash)*H*T;
H = H/H(3,3);

%a test
pTest = H*pointsR(1,:)';
pTest = pTest / pTest(3);
disp(sprintf('Mean geometric error for the first point: %d', abs(mean(pointsL(1,:)' - pTest)))); 
end

%carry all points around 0,0 with mean euclidean distance of sqrt(2)
function T = NormalizePoints(points)
    meanP = mean(points);
    T = [1 0 -meanP(1);0 1 -meanP(2);0 0 1];
    nL = T*points';
    sL = mean(abs(nL'));
    S = [1/sL(1) 0 0; 0 1/sL(2) 0; 0 0 1];
    T = S * T;
end

