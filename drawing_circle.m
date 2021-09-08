clc
clear all

nx=400;
ny=400;
space = zeros(ny,nx);
r=20;
time_step=4*r
new_x(1)=150;
new_y(1)=150;
N=10;
center_x=new_x(1);
center_y=new_y(1)-r;



% for i = 2:time_step
%     if new_x(i-1) < center_x && new_x(i-1)>center_x-r && new_y(i-1) < center_y
%         new_x(i)=new_x(i-1)-1;
%         new_y(i) = -sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) < center_x && new_x(i-1)> center_x-r && new_y(i-1) > center_y
%         new_x(i)=new_x(i-1)+1;
%         new_y(i) = sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) > center_x && new_x(i-1) < center_x+r && new_y(i-1) > center_y
%         new_x(i)=new_x(i-1)+1;
%         new_y(i) = sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) > center_x && new_x(i-1)< center_x+r && new_y(i-1) < center_y
%         new_x(i)=new_x(i-1)-1;
%         new_y(i) = -sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) == center_x && new_y(i-1) == center_y-r
%         new_x(i)=new_x(i-1)-1;
%         new_y(i) = -sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) == center_x && new_y(i-1) == center_y+r
%         new_x(i)=new_x(i-1)+1;
%         new_y(i) = sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) == center_x-r && new_y(i-1) == center_y
%         new_x(i)=new_x(i-1)+1;
%         new_y(i) = sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     elseif new_x(i-1) == center_x+r && new_y(i-1) == center_y
%         new_x(i)=new_x(i-1)-1;
%         new_y(i) = -sqrt(r^2-(new_x(i)-center_x)^2)+center_y;
%     end
%     
% end
%initialize agents, random locations
new_x=zeros(N,time_step);
new_y=zeros(N,time_step);

new_x(:,1) = randi([100,300],N,1);
new_y (:,1)= randi([100,300],N,1);

for i = 1:N
    RR(i)= (-1/200)*(new_x(i,1))^2+2*(new_x(i,1))-150;
    R(i)=round(RR(i));
end

time_step=4*max(R);




for i=1:N
    center_x(i)=new_x(i,1);
    center_y(i)=new_y(i,1)-R(i);
end

for i = 1: N
    for j = 2:time_step
    if new_x(i,j-1) < center_x(i) && new_x(i,j-1)>center_x(i)-R(i) && new_y(i,j-1) < center_y(i)
        new_x(i,j)=new_x(i,j-1)-1;
        new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
       
    elseif new_x(i,j-1) < center_x(i) && new_x(i,j-1)> center_x(i)-R(i) && new_y(i,j-1) > center_y(i)
        new_x(i,j)=new_x(i,j-1)+1;
        new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
     
    elseif new_x(i,j-1) > center_x(i) && new_x(i,j-1) < center_x(i)+R(i) && new_y(i,j-1) > center_y(i)
        new_x(i,j)=new_x(i,j-1)+1;
        new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
  
    elseif new_x(i,j-1) > center_x(i) && new_x(i,j-1)< center_x(i)+R(i) && new_y(i,j-1) < center_y(i)
        new_x(i,j)=new_x(i,j-1)-1;
        new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
       
    elseif new_x(i,j-1) == center_x(i) && new_y(i,j-1) == center_y(i)-R(i)
        new_x(i,j)=new_x(i,j-1)-1;
        new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
   
    elseif new_x(i,j-1) == center_x(i) && new_y(i,j-1) == center_y(i)+R(i)
        new_x(i,j)=new_x(i,j-1)+1;
        new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
        
    elseif new_x(i,j-1) == center_x(i)-R(i) && new_y(i,j-1) == center_y(i)
        new_x(i,j)=new_x(i,j-1)+1;
        new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
 
    elseif new_x(i,j-1) == center_x(i)+R(i) && new_y(i,j-1) == center_y(i)
        new_x(i,j)=new_x(i,j-1)-1;
        new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
       
    end
    end
end


% for i = 1: N
%     for j = 2:time_step
%     if new_x(i,j-1) < center_x(i) && new_x(i,j-1)>center_x(i)-R(i) && new_y(i,j-1) < center_y(i)
%         new_x(i,j)=new_x(i,j-1)-1;
%         new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%        
%     elseif new_x(i,j-1) < center_x(i) && new_x(i,j-1)> center_x(i)-R(i) && new_y(i,j-1) > center_y(i)
%         new_x(i,j)=new_x(i,j-1)+1;
%         new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%      
%     elseif new_x(i,j-1) > center_x(i) && new_x(i,j-1) < center_x(i)+R(i) && new_y(i,j-1) > center_y(i)
%         new_x(i,j)=new_x(i,j-1)+1;
%         new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%   
%     elseif new_x(i,j-1) > center_x(i) && new_x(i,j-1)< center_x(i)+R(i) && new_y(i,j-1) < center_y(i)
%         new_x(i,j)=new_x(i,j-1)-1;
%         new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%        
%     elseif new_x(i,j-1) == center_x(i) && new_y(i,j-1) == center_y(i)-R(i)
%         new_x(i,j)=new_x(i,j-1)-1;
%         new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%    
%     elseif new_x(i,j-1) == center_x(i) && new_y(i,j-1) == center_y(i)+R(i)
%         new_x(i,j)=new_x(i,j-1)+1;
%         new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%         
%     elseif new_x(i,j-1) == center_x(i)-R(i) && new_y(i,j-1) == center_y(i)
%         new_x(i,j)=new_x(i,j-1)+1;
%         new_y(i,j) = sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%  
%     elseif new_x(i,j-1) == center_x(i)+R(i) && new_y(i,j-1) == center_y(i)
%         new_x(i,j)=new_x(i,j-1)-1;
%         new_y(i,j) = -sqrt((R(i))^2-(new_x(i,j)-center_x(i))^2)+center_y(i);
%        
%     end
%     end
% end

% for i=1:N
%     for j=1:time_step
%         space(new_y(i,j),new_x(i,j))=1;
%         %space(locs_delete{j}(1),locs_delete{j}(2))=0;
%     end
% end



new_y=round(new_y)
for i =1:N
    for j=1:time_step
    space(new_y(i,j),new_x(i,j))=1;
    end
end

figure; imshow(space);