function [IOut2,output] = denoise_PDL_compare2(Image,sigma,atomnum,varargin) 
%  Function denoise_PDL_compare2.m is the core£¬where lines 80-83 are the iterative equations in lines 4-5 of Diagram2¡¾2¡¿
%  
% ¡¾1¡¿Qiegen Liu, Shanshan Wang, Jianhua Luo. A novel predual dictionary learning algorithm. Journal of Visual Communication and Image Representation, 2012, 23(1): 182-193. 
% ¡¾2¡¿Qiegen Liu, Jianhua Luo, Shanshan Wang, Moyan Xiao and Meng Ye. An augmented Lagrangian Multi-scale Dictionary Learning Algorithm. EURASIP Journal on Advances in Signal Processing 2011, 2011:58 doi:10.1186/1687-6180-2011-58.
%

reduceDC = 1;
[NN1,NN2] = size(Image); 
waitBarOn = 0; 
displaydic = 0;
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
    if (strcmp(varargin{argI}, 'displayFlag')) 
        displayFlag = varargin{argI+1}; 
    end
    if (strcmp(varargin{argI}, 'displaydic')) 
        displaydic = varargin{argI+1}; 
    end
    if (strcmp(varargin{argI}, 'waitBarOn')) 
        waitBarOn = varargin{argI+1}; 
    end 
    if (strcmp(varargin{argI}, 'blkMatrix_yes')) 
        blkMatrix = varargin{argI+1}; 
    end
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = initialDictionary;
[atomsize, samplenum] = size(blkMatrix);
C = 1.15;
errT        =   C*sigma;   % 0.1
nbIter      =   600;       %160 or  other initializations
tau         =   errT*sqrt(atomsize);
Alpha       =   145;        %[0.001 0.01 0.1 1 10 100 1000];
C_MM        =   2.00;
lambda_grad =   0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf('tau: %f, nbIter: %d, Alpha: %f, errT: %f \n',tau,nbIter,Alpha,errT);
Anorm       =   get_operator_norm_Liu(A,0);
%Anorm       =   8;
rho_without_alpha = 1./(Anorm*Anorm);
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%% initialization
init_x = ones(atomnum,samplenum);
init_x = abs(randn(atomnum,samplenum)).^3;
init_x = init_x/sqrt(mean(mean(init_x.^2)));
x = init_x;
%x = zeros(atomnum,samplenum);
cm = init_c_dictionary(A,blkMatrix,x);
[dk,residual_norm,Ax] = compute_dk_dictionary(A,blkMatrix,x,cm,Alpha,tau);
%%%%%%%%%%%%%%%%%%%%%%%%%% main loop
time=1;m=0;stop_end = 0;
while (time < nbIter),  %loop in m
    %%%  sparse coding
    k=0;
    while  ( k<7 & time<nbIter), % loop in k normEvol > 10/128/50&
        Atdk = A' * dk;
%         tic;
%         x = sparse(x);
        x = x+rho_without_alpha*Alpha*C_MM*Atdk;      
%         x = sparse(x);
        x = soft(x,rho_without_alpha*Alpha*C_MM);
        [dk,residual_norm,Ax] = compute_dk_dictionary(A,blkMatrix,sparse(x),cm,Alpha,tau);
%         toc;    
        if sigma<50
            if residual_norm <= errT/1.11,   stop_end = 1; end%break;            
        elseif sigma<=75
            if residual_norm <= errT/1.15,   stop_end = 1; end%break;       
        else
            if residual_norm <= errT/1.122,   stop_end = 1; end%break;
        end        
        k=k+1;
        time=time+1;
    end;  % end loop in k
    if sigma>=75
        if m>=10,  stop_end = 1;   end  %18
    elseif sigma>=50
        if m>=10,  stop_end = 1;   end  %18
    else        
        if m>=24,  stop_end = 1;   end  %18
    end    
    if stop_end == 1;  break;   end

    %%%  dictionary updating
    %A = A + lambda_grad*dk*x';
    A = A + dk*x'*diag(diag(x*x' + 1e-7*speye(size(x,1))));  %lambda_grad*
    %max(max(diag(diag(x*x'))))
    A = A ./ repmat( sqrt(sum(A.^2,1)), atomsize,1 );
    Anorm = get_operator_norm_Liu(A,0);
    rho_without_alpha = 1./(Anorm*Anorm);
    if displaydic == 1
        figure(197);II = displayDictionary_nonsquare2(A,0); pause(0.1);%+m
    end
    m=m+1;
    cm=dk;
    Alpha = Alpha/1.04; 
    %if m>15,  Alpha =75;   end   %default
    if m>23,  Alpha =58;   end
    fprintf('m: %d , residual_norm: %5.2f, Alpha: %5.2f\n', m, residual_norm, Alpha);    
    
end;  % end loop in m

[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);  clear blkMatrix;
Dictionary = A; clear A;
output.D = Dictionary; 

C = 1.15;
errT = C*sigma;
% go with jumps of 30000
for jj = 1:30000:size(blocks,2)
    jumpSize = min(jj+30000-1,size(blocks,2));
    if (reduceDC)
        vecOfMeans = mean(blocks(:,jj:jumpSize));
        blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    end
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
toc;

IOut2 = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight); 
%IOut2 = (Image+10*sigma*IMout/255)./(1+10*sigma*Weight/255); 
%IOut = (0+0.034*sigma*IMout)./(0+0.034*sigma*Weight); 

