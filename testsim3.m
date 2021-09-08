% Simple multiagent simulator
% v. 0.1
% Date 8/19/2021
% Jordan R. Raney
% modified v. 0.2 (add random_deposition_deletion_rules)
% Date 8/20/2021
clear nx ny N tmax space locs randxvals randyvals movecandidates adjmater r2extcounter;

%General parameters (relevant to all rules)
nx = 400; % width of space
ny = 400; % height of space
N = 12; % number of agents
N_delete=5; %number of agents for materials removal
tmax = 80000; % max time, i.e., number of iterations
movetm = 1; % 1 means the agents can move through material, 0 means they cannot
mu_x=nx/2; % normal distribution parameters
mu_y=ny/2;
sigma_x=nx;
sigma_y=ny;
%Choose a rule
%Rule 1: Brownian motion: All agents extrude while moving randomly.
%Rule 2: All agents extrude if they contact pre-existing material. Then move randomly.
rule=2;

%Rule-specific parameters:
r2ep = 50; % Rule 2 "extrusion period". The number of iterations to extrude for once extrusion starts.
r2dp = 10; % Rule 2 "dead period". The number of iterations agents should wait before extruding again.

% initialize space
space = zeros(ny,nx);

% add border material around outer edge of space
space(1,:)=1;
space(ny,:)=1;
space(:,1)=1;
space(:,nx)=1;


%gaussian distribution process
randxvals= normrnd(mu_x,sigma_x,[N,N]);
randyvals = normrnd(mu_y,sigma_y,[N,N]);
randxvals_delete = normrnd(mu_x,sigma_x,[N_delete,N_delete]);
randyvals_delete = normrnd(mu_y,sigma_y,[N_delete,N_delete]);



for p=1:N %%check agents are in the range of size
    for pp=1:N
        randxvals(p,pp)=round(randxvals(p,pp));
        randyvals(p,pp)=round(randyvals(p,pp));
        if randxvals(p,pp)<2
            randxvals(p,pp)=2;
        elseif randxvals(p,pp)>nx-2
            randxvals(p,pp)=nx-2;
        end
        if randyvals(p,pp)<2
             randyvals(p,pp)=2;
        elseif randyvals(p,pp)>ny-2
            randyvals(p,pp)=ny-2;
        end
    end
end

for p=1:N_delete
    for pp=1:N_delete
        randxvals_delete(p,pp)=round(randxvals_delete(p,pp));
        randyvals_delete(p,pp)=round(randyvals_delete(p,pp));
        if randxvals_delete(p,pp)<2
            randxvals_delete(p,pp)=2;
        elseif randxvals_delete(p,pp)>nx-2
            randxvals_delete(p,pp)=nx-2;
        end
        if randyvals_delete(p,pp)<2
             randyvals_delete(p,pp)=2;
        elseif randyvals_delete(p,pp)>ny-2
            randyvals_delete(p,pp)=ny-2;
        end
    end
end

%initialize agents, random locations
% randxvals = randi([2,nx-1],N);
% randyvals = randi([2,ny-1],N);
% randxvals_delete = randi([2,nx-1],N_delete);
% randyvals_delete = randi([2,ny-1],N_delete);
for i=1:N
    locs{i}=[randyvals(i) randxvals(i)];
    locs_delete{i}=[randyvals_delete(i) randxvals_delete(i)];
end
for i=1:N_delete
locs_delete{i}=[randyvals_delete(i) randxvals_delete(i)];
end


r2extcounter=zeros(N,1)+r2ep; % this keeps track of whether or not each agent is allowed to extrude (based on rule 2)
% begin simulation
final_space=random_deposition_deletion(rule,space,r2extcounter,r2ep,r2dp,movetm,tmax,N,locs,nx,ny,locs_delete,N_delete);

% identify final locations of the agents in gray
for jx=1:N
    space(locs{jx}(1), locs{jx}(2))=0.5;
end

% plot the result
figure; imshow(final_space);
movement_delete=randi([1 4]);

function final_space = random_deposition_deletion(rule,space,r2extcounter,r2ep,r2dp,movetm,tmax,N,locs,nx,ny,locs_delete,N_delete)
for ii = 1:tmax
    for k=1:N
        adjmater=zeros(1,4);
        if space(locs{k}(1),locs{k}(2))~=1 % no need to extrude if there is already material
             if locs{k}(1)>1
                 adjmater(1)=space(locs{k}(1)-1, locs{k}(2)); % check up for material
             end
             if locs{k}(2)<nx
                 adjmater(2)=space(locs{k}(1), locs{k}(2)+1); % check right for material
             end
             if locs{k}(1)<ny
                 adjmater(3)=space(locs{k}(1)+1, locs{k}(2)); % check down for material
             end
             if locs{k}(2)>1
                 adjmater(4)=space(locs{k}(1), locs{k}(2)-1); % check left for material
             end
         end
         if sum(adjmater)>0 %if true, there is material adjacent to the current location, so extrude
             if r2extcounter(k)>0 % r2extcounter must be greater than 0 to allow the agent to extrude
                 space(locs{k}(1),locs{k}(2))=1; %extrude
                 r2extcounter(k)=r2extcounter(k)-1; %the agent uses up one of its extrusion iterations
             end
         end
            % update the extrusion counter to determine whether the agent is allowed to extrude next time
         if r2extcounter(k)==0
             r2extcounter(k)=-1*r2dp-1; %next time it's not allowed to extrude. set to negative value, increment each iteration as you wait.
         elseif r2extcounter(k)<-1
             r2extcounter(k)=r2extcounter(k)+1; %wait another iteration
         elseif r2extcounter(k)==-1
             r2extcounter(k)=r2ep; %next time it's allowed to extrude again
         end
    
        % find available directions to potentially move next
        movecandidates = zeros(1,4);
        
        % check up
        if locs{k}(1)-1>0
            if movetm==0
                if space(locs{k}(1)-1,locs{k}(2))==0
                    movecandidates(1)=1; %it's ok to move up
                end
            else
                movecandidates(1)=1; %it's ok to move up
            end
        end
        
        % check right
        if locs{k}(2)+1<=nx
            if movetm==0
                if space(locs{k}(1),locs{k}(2)+1)==0
                    movecandidates(2)=1; %it's ok to move right
                end
            else
                movecandidates(2)=1; %it's ok to move right
            end
        end
        
        % check down
        if locs{k}(1)+1<=ny
            if movetm==0
                if space(locs{k}(1)+1,locs{k}(2))==0
                    movecandidates(3)=1; %it's ok to move down
                end
            else
                movecandidates(3)=1; %it's ok to move down
            end
        end
        
        % check left
        if locs{k}(2)-1>0
            if movetm==0
                if space(locs{k}(1),locs{k}(2)-1)==0
                    movecandidates(4)=1; %it's ok to move left
                end
            else
                movecandidates(4)=1; %it's ok to move left
            end
        end
        
        l=0;
        if sum(movecandidates)>0
            % choose an available direction
            chooserand=randi([1,sum(movecandidates)]);
            for kk=1:chooserand
                l=l+1;
                while movecandidates(l)==0
                    l = l+1;
                end
            end
        end
        % else there is no direction available, so the agent doesn't move
        
        % set the next location
        if l==1
            locs{k}(1)=locs{k}(1)-1; %move up       
        elseif l==2
            locs{k}(2)=locs{k}(2)+1; %move right
        elseif l==3
            locs{k}(1)=locs{k}(1)+1; %move down
        elseif l==4
            locs{k}(2)=locs{k}(2)-1; %move left
        end
       
    end
    
    for j = 1:N_delete
        ff=space(locs_delete{j}(1),locs_delete{j}(2));
        
        if ff==1
            space(locs_delete{j}(1),locs_delete{j}(2))=0; %%material removal
        end
        
        movement_delete(j,ii)=randi([1 4]);
        if movement_delete(j,ii)==1
            if locs_delete{j}(1)>=5
                locs_delete{j}(1)=locs_delete{j}(1)-1;
            else locs_delete{j}(1)=locs_delete{j}(1)+1;
            end
        end
        if movement_delete(j,ii)==2
            if locs_delete{j}(2)<=nx-5
                locs_delete{j}(2)=locs_delete{j}(2)+1;
            else locs_delete{j}(2)=locs_delete{j}(2)-1;
            end
        end
        if movement_delete(j,ii)==3
            if locs_delete{j}(1)<=ny-5
                locs_delete{j}(1)=locs_delete{j}(1)+1;
            else locs_delete{j}(1)=locs_delete{j}(1)-1;
            end
        end
        if movement_delete(j,ii)==4
            if locs_delete{j}(2)>=5
                locs_delete{j}(2)=locs_delete{j}(2)-1;
            else locs_delete{2}(2)=locs_delete{j}(2)+1;
            end
        end
    end
end 
final_space = space;
end




% function final_space = random_deposition(rule,space,r2extcounter,r2ep,r2dp,movetm,tmax,N,locs,nx,ny)
% for i=1:tmax
%     
%     if rule==1
%         % Rule 1: All agents extrude then move in a random direction.
%         for j=1:N
%             space(locs{j}(1),locs{j}(2))=1; % extrude where you are
%         end
%     if rule==2
%         %Rule 2: All agents extrude if they contact pre-existing material. Then move randomly.
%         for j=1:N
%             adjmater=zeros(1,4);
%             if space(locs{j}(1),locs{j}(2))~=1 % no need to extrude if there is already material
%                 if locs{j}(1)>1
%                     adjmater(1)=space(locs{j}(1)-1, locs{j}(2)); % check up for material
%                 end
%                 if locs{j}(2)<nx
%                     adjmater(2)=space(locs{j}(1), locs{j}(2)+1); % check right for material
%                 end
%                 if locs{j}(1)<ny
%                     adjmater(3)=space(locs{j}(1)+1, locs{j}(2)); % check down for material
%                 end
%                 if locs{j}(2)>1
%                     adjmater(4)=space(locs{j}(1), locs{j}(2)-1); % check left for material
%                 end
%             end
%             if sum(adjmater)>0 %if true, there is material adjacent to the current location, so extrude
%                 if r2extcounter(j)>0 % r2extcounter must be greater than 0 to allow the agent to extrude
%                     space(locs{j}(1),locs{j}(2))=1; %extrude
%                     r2extcounter(j)=r2extcounter(j)-1; %the agent uses up one of its extrusion iterations
%                 end
%             end
%             % update the extrusion counter to determine whether the agent is allowed to extrude next time
%             if r2extcounter(j)==0
%                 r2extcounter(j)=-1*r2dp-1; %next time it's not allowed to extrude. set to negative value, increment each iteration as you wait.
%             elseif r2extcounter(j)<-1
%                 r2extcounter(j)=r2extcounter(j)+1; %wait another iteration
%             elseif r2extcounter(j)==-1
%                 r2extcounter(j)=r2ep; %next time it's allowed to extrude again
%             end
%         end
%     end
%      
%     
%     % Movement for next iteration:
%     for j=1:N
%         % find available directions to potentially move next
%         movecandidates = zeros(1,4);
%         
%         % check up
%         if locs{j}(1)-1>0
%             if movetm==0
%                 if space(locs{j}(1)-1,locs{j}(2))==0
%                     movecandidates(1)=1; %it's ok to move up
%                 end
%             else
%                 movecandidates(1)=1; %it's ok to move up
%             end
%         end
%         
%         % check right
%         if locs{j}(2)+1<=nx
%             if movetm==0
%                 if space(locs{j}(1),locs{j}(2)+1)==0
%                     movecandidates(2)=1; %it's ok to move right
%                 end
%             else
%                 movecandidates(2)=1; %it's ok to move right
%             end
%         end
%         
%         % check down
%         if locs{j}(1)+1<=ny
%             if movetm==0
%                 if space(locs{j}(1)+1,locs{j}(2))==0
%                     movecandidates(3)=1; %it's ok to move down
%                 end
%             else
%                 movecandidates(3)=1; %it's ok to move down
%             end
%         end
%         
%         % check left
%         if locs{j}(2)-1>0
%             if movetm==0
%                 if space(locs{j}(1),locs{j}(2)-1)==0
%                     movecandidates(4)=1; %it's ok to move left
%                 end
%             else
%                 movecandidates(4)=1; %it's ok to move left
%             end
%         end
%         
%         l=0;
%         if sum(movecandidates)>0
%             % choose an available direction
%             chooserand=randi([1,sum(movecandidates)]);
%             for k=1:chooserand
%                 l=l+1;
%                 while movecandidates(l)==0
%                     l = l+1;
%                 end
%             end
%         end
%         % else there is no direction available, so the agent doesn't move
%         
%         % set the next location
%         if l==1
%             locs{j}(1)=locs{j}(1)-1; %move up       
%         elseif l==2
%             locs{j}(2)=locs{j}(2)+1; %move right
%         elseif l==3
%             locs{j}(1)=locs{j}(1)+1; %move down
%         elseif l==4
%             locs{j}(2)=locs{j}(2)-1; %move left
%         end
%     end
% end
% final_space = space;
% end
% end
