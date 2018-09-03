%% The Code is created based on the method described in the following papers: 
%  A*x = blkMatrix    learning dictionary
%  Dictionary*Coefs = blocks   sparse coding under learnd dictionary
%  This idea was originally motivated from the predual dictionary learning (PDL)¡¾1¡¿
%  Later, we found that it also can be derived from the view of augmented Lagrangian¡¾2¡¿
% ¡¾2¡¿is the advanced version, termed ALM-DL:augmented Lagrangian multi-scale dictionary learning
%  Function denoise_PDL_compare2.m is the core£¬where lines 80-83 are the iterative equations in lines 4-5 of Diagram2¡¾2¡¿
%  Version : 2.0   
% Please cite the following papers when you use th code
% ¡¾1¡¿Qiegen Liu, Shanshan Wang, Jianhua Luo. A novel predual dictionary learning algorithm. Journal of Visual Communication and Image Representation, 2012, 23(1): 182-193. 
% ¡¾2¡¿Qiegen Liu, Jianhua Luo, Shanshan Wang, Moyan Xiao and Meng Ye. An augmented Lagrangian Multi-scale Dictionary Learning Algorithm. EURASIP Journal on Advances in Signal Processing 2011, 2011:58 doi:10.1186/1687-6180-2011-58.
%
% All rights reserved.
% This work should only be used for nonprofit purposes.
%

clear all; close all; clc; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
getd = @(p)path(path,p);% Add some directories to the path
getd('demo_ulities\');

%%%%%%%########### load the reference and noisy image (M0/M) #######
sigma = 15;  
noise = sigma*randn(256,256); 
pathForImages ='';
name_list = {'barbara256.png','house256.png','Boat256.jpg','Lena256.png','peppers256.png','cameraman256.png' };
nbimg = length(name_list);
psnrr = zeros(nbimg,2);
ssim = zeros(nbimg,2);
PSNRIn = 0; PSNROut1 = 0; PSNROut3 = 0; SSIM1 = 0; SSIM3 = 0;
for nbimgi = 1:1
    [M0,pp] = imread(strcat([pathForImages,name_list{nbimgi}]));
    M0 = im2double(M0);
    if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
    if (max(M0(:))<2);   M0 = M0*255;    end
    M=M0 + noise;    
    %PSNRIn = 20*log10(255/sqrt(mean((M(:)-M0(:)).^2)));  
    %%%%%%%########### extract image patches #######
    [NN1,NN2] = size(M);
    bb = 8;
    slidingDis = 2;
    reduceDC = 1;
    [blkMatrix,idx] = my_im2col(M,[bb,bb],slidingDis);

    if (reduceDC)
        vecOfMeans = mean(blkMatrix);
        blkMatrix = blkMatrix-ones(size(blkMatrix,1),1)*vecOfMeans;
    end
    samplenum = size(blkMatrix,2);
    %%%%%%%########### initialize the dictionary #######
    atomnum = 2*bb^2; % number of atoms in the dictionary
    dictionary = 'DCT';
    switch dictionary
        case 'input'
            sel = randperm(samplenum); sel = sel(1:atomnum);
            A = blkMatrix(:,sel);
            A = A ./ repmat( sqrt(sum(A.^2)), [bb^2, 1] );
        case 'DCT'
            Pn = ceil(sqrt(atomnum));
            DCT=zeros(bb,Pn);
            for k=0:1:Pn-1,
                V=cos([0:1:bb-1]'*k*pi/Pn);
                if k>0, V=V-mean(V); end;
                DCT(:,k+1)=V/norm(V);
            end;
            A = kron(DCT,DCT);            
            A = A(:,1:atomnum);            
    end    
%     figure(466); imshow(M,[]);
%     figure(566); II = displayDictionary_nonsquare2(A,0);  %title('The initial dictionary ');           
      
    [atomsize, atomnum] = size(A);
    [atomsize, samplenum] = size(blkMatrix);
    %---------  for compare purposes ------------------------
    fprintf('blocksize: %d , slidingFactor: %d, sigma: %d \n', bb, slidingDis, sigma);
    fprintf('atomsize : %d, atomnum: %d,  samplenum: %d \n',atomsize, atomnum, samplenum);
    displaydic = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%           PDL             %%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [resultM1,output] = ...
        denoise_PDL_compare2(M,sigma,atomnum,'slidingFactor',slidingDis,'blockSize',bb,'initialDictionary',A,'blkMatrix_yes',blkMatrix,'displaydic',displaydic);
    
    PSNROut1 = 20*log10(255/sqrt(mean((resultM1(:)-M0(:)).^2)))
    SSIM1 = ssim_index(resultM1,M0);
    
    figure(477);   imshow(resultM1,[]);  
    figure(577);    III = displayDictionary_nonsquare2(output.D,0);    
    %return;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%         K-SVD             %%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [resultM3,output] = ...
        denoise_KSVD_compare2(M,sigma,atomnum,'slidingFactor',slidingDis,'blockSize',bb,'initialDictionary',A,'blkMatrix_yes',blkMatrix,'displaydic',displaydic);
    toc;
    PSNROut3 = 20*log10(255/sqrt(mean((resultM3(:)-M0(:)).^2)))
    SSIM3 = ssim_index(resultM3,M0);
    
    figure(499);   imshow(resultM3,[]);  
    figure(599);    III = displayDictionary_nonsquare2(output.D,0);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%        display            %%%%%%%%%%%%%%%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [PSNROut1, PSNROut3, SSIM1, SSIM3]
    psnrr(nbimgi,:) = [PSNROut1, PSNROut3];   
    ssim(nbimgi,:) = [SSIM1, SSIM3];  
end
psnrr'
ssim'