function [IOut2,output] = denoise_KSVD_compare2(Image,sigma,K,varargin) 
% first, train a dictionary on the noisy image 
 
reduceDC = 1; 
[NN1,NN2] = size(Image); 
waitBarOn = 1; 
if (sigma > 5) 
    numIterOfKsvd = 10; 
else 
    numIterOfKsvd = 5; 
end 
C = 1.15; 
displayFlag = 1; 
displaydic = 0;
maxBlocksToConsider = 260000; 
maxNumBlocksToTrainOn = 65000; 
 
for argI = 1:2:length(varargin) 
    if (strcmp(varargin{argI}, 'slidingFactor')) 
        slidingDis = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'initialDictionary')) 
        initialDictionary = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'errorFactor')) 
        C = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'maxBlocksToConsider')) 
        maxBlocksToConsider = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'numKSVDIters')) 
        numIterOfKsvd = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'blockSize')) 
        bb = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'maxNumBlocksToTrainOn')) 
        maxNumBlocksToTrainOn = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'displaydic')) 
        displaydic = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'displayFlag')) 
        displayFlag = varargin{argI+1}; 
    end
    if (strcmp(varargin{argI}, 'waitBarOn')) 
        waitBarOn = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'blkMatrix_yes')) 
        blkMatrix = varargin{argI+1}; 
    end
end 
 
% first, train a dictionary on blocks from the noisy image 
 
% if(prod([NN1,NN2]-bb+1)> maxNumBlocksToTrainOn) 
%     randPermutation =  randperm(prod([NN1,NN2]-bb+1)); 
%     selectedBlocks = randPermutation(1:maxNumBlocksToTrainOn); 
%  
%     blkMatrix = zeros(bb^2,maxNumBlocksToTrainOn); 
%     for i = 1:maxNumBlocksToTrainOn 
%         [row,col] = ind2sub(size(Image)-bb+1,selectedBlocks(i)); 
%         currBlock = Image(row:row+bb-1,col:col+bb-1); 
%         blkMatrix(:,i) = currBlock(:); 
%     end 
% else 
%     %blkMatrix = im2col(Image,[bb,bb],'sliding'); 
%     [blkMatrix,idx1] = my_im2col(Image,[bb,bb],slidingDis); 
% end 
blkMatrix = blkMatrix;
 
param.K = K; 
param.numIteration = numIterOfKsvd ; 
 
param.errorFlag = 1; % decompose signals until a certain error is reached. do not use fix number of coefficients. 
param.errorGoal = sigma*C; 
param.preserveDCAtom = 0; 
 
% Pn=ceil(sqrt(K)); 
% DCT=zeros(bb,Pn); 
% for k=0:1:Pn-1, 
%     V=cos([0:1:bb-1]'*k*pi/Pn); 
%     if k>0, V=V-mean(V); end; 
%     DCT(:,k+1)=V/norm(V); 
% end; 
% DCT=kron(DCT,DCT); 
% %%%%%%%%%%%%%%%%  add by me %%%%%%%%%
% DCT=DCT(:,1:K);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.initialDictionary = initialDictionary; 
param.InitializationMethod =  'GivenMatrix'; 
 
% if (reduceDC) 
%     vecOfMeans = mean(blkMatrix); 
%     blkMatrix = blkMatrix-ones(size(blkMatrix,1),1)*vecOfMeans; 
% end 
 
if (waitBarOn) 
    counterForWaitBar = param.numIteration+1; 
    h = waitbar(0,'Denoising In Process ...'); 
    param.waitBarHandle = h; 
    param.counterForWaitBar = counterForWaitBar; 
end 
 
 
param.displayProgress = displayFlag; 

if displaydic == 0
    [Dictionary,output] = KSVD(blkMatrix,param); 
else
    [Dictionary,output] = KSVD_Liu_display(blkMatrix,param);
end
output.D = Dictionary; 
 
if (displayFlag) 
    disp('finished Trainning dictionary'); 
end 
 
 
% denoise the image using the resulted dictionary 
errT = sigma*C; 
IMout=zeros(NN1,NN2); 
Weight=zeros(NN1,NN2); 
%blocks = im2col(Image,[NN1,NN2],[bb,bb],'sliding'); 
while (prod(floor((size(Image)-bb)/slidingDis)+1)>maxBlocksToConsider) 
    slidingDis = slidingDis+1; 
end 
[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis); 
 
if (waitBarOn) 
    newCounterForWaitBar = (param.numIteration+1)*size(blocks,2); 
end 
 
 
% go with jumps of 30000 
for jj = 1:30000:size(blocks,2) 
    if (waitBarOn) 
        waitbar(((param.numIteration*size(blocks,2))+jj)/newCounterForWaitBar); 
    end 
    jumpSize = min(jj+30000-1,size(blocks,2)); 
    if (reduceDC) 
        vecOfMeans = mean(blocks(:,jj:jumpSize)); 
        blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1); 
    end 
     
    %Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),Dictionary,errT); 
    Coefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),errT); 
    if (reduceDC) 
        blocks(:,jj:jumpSize)= Dictionary*Coefs + ones(size(blocks,1),1) * vecOfMeans; 
    else 
        blocks(:,jj:jumpSize)= Dictionary*Coefs ; 
    end 
end 
 
count = 1; 
Weight = zeros(NN1,NN2); 
IMout = zeros(NN1,NN2); 
[rows,cols] = ind2sub(size(Image)-bb+1,idx); 
for i  = 1:length(cols) 
    col = cols(i); row = rows(i);         
    block =reshape(blocks(:,count),[bb,bb]); 
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block; 
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb); 
    count = count+1; 
end; 
 
if (waitBarOn) 
    close(h); 
end 
IOut2 = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight); 
%IOut2 = (Image+10*sigma*IMout/255)./(1+10*sigma*Weight/255); 
%IOut = (0+0.034*sigma*IMout)./(0+0.034*sigma*Weight); 
