clc;
clear all;

N = 10; M = 10;                  %# grid size
d = 'cityblock';
[X Y] = meshgrid(1:N,1:M);
X = X(:); Y = Y(:);
admatrix = squareform( pdist([X Y], d) == 1 );
[x y] = gplot(admatrix, [X Y]);
subplot(2,1,1); 
plot(x, y, 'ks-', 'MarkerFaceColor','r');
Step=20;
sensorX=zeros(1,Step);
sensorY=zeros(1,Step);

sensorX(:,1)=5;
sensorY(:,1)=5;


 for i=2:Step
r1(i)=randi([1, 2], 1);
r2(i)=randi([1, 2], 1);
if r1(i)==1 && r2(i)==1
     if sensorX(:,i-1)<N
    sensorX(:,i)=sensorX(:,i-1)+1;
    sensorY(:,i)=sensorY(:,i-1);
     end
   if sensorX(:,i-1)==N && sensorY(:,i-1)<N 
        sensorX(:,i)=sensorX(:,i-1)
       sensorY(:,i)=sensorY(:,i-1)+1; 
   end
    if sensorX(:,i-1)==N && sensorY(:,i-1)==N 
          sensorX(:,i)=sensorX(:,i-1)
       sensorY(:,i)=sensorY(:,i-1)-1; 
    end
end
 if r1(i)==1 && r2(i)==2
     if sensorX(:,i-1)>1
    sensorX(:,i)=sensorX(:,i-1)-1; 
    sensorY(:,i)=sensorY(:,i-1);
     end
    if sensorX(:,i-1)==1 && sensorY(:,i-1)<N 
        sensorX(:,i)=sensorX(:,i-1)
       sensorY(:,i)=sensorY(:,i-1)+1; 
    end
     if sensorX(:,i-1)==1 && sensorY(:,i-1)==N 
        sensorX(:,i)=sensorX(:,i-1)
       sensorY(:,i)=sensorY(:,i-1)-1; 
    end
 end
 if r1(i)==2 && r2(i)==1
     if sensorY(:,i-1)<N
    sensorY(:,i)=sensorY(:,i-1)+1;
    sensorX(:,i)=sensorX(:,i-1)
     end
 if sensorY(:,i-1)==N && sensorX(:,i-1)<N 
        sensorY(:,i)=sensorY(:,i-1)
       sensorX(:,i)=sensorX(:,i-1)+1; 
 end
   if sensorY(:,i-1)==N && sensorX(:,i-1)==N 
        sensorY(:,i)=sensorY(:,i-1)
       sensorX(:,i)=sensorX(:,i-1)-1; 
   end
 end
 if r1(i)==2 && r2(i)==2
  
     if sensorY(:,i-1)>1
    sensorY(:,i)=sensorY(:,i-1)-1;
    sensorX(:,i)=sensorX(:,i-1);
     end
    if sensorY(:,i-1)==1 && sensorX(:,i-1)<N 
        sensorY(:,i)=sensorY(:,i-1)
       sensorX(:,i)=sensorX(:,i-1)+1; 
    end
    if sensorY(:,i-1)==1 && sensorX(:,i-1)==N 
        sensorY(:,i)=sensorY(:,i-1)
       sensorX(:,i)=sensorX(:,i-1)-1; 
    end
 end 
 end
 subplot(2,1,2);
plot(sensorX,sensorY, 'ks-', 'MarkerFaceColor','r');

!git add file1.m
!git commit -m "added file1.m"
!git push

    