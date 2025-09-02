clc; clear all

%%%%%%%%%%% VISUALIZZAZIONI SERIE STORICHE %%%%%%%%%%%
%ripple
XRP = readmatrix("XRP.xlsx")
XRPclose = XRP(:,2)
pXRP = log(XRPclose)
yXRP = 100*(diff(pXRP))

%litecoin
LTC = readmatrix("LTC .xlsx")
LTCclose = LTC(:,2)
pLTC = log(LTCclose)
yLTC = 100*(diff(pLTC))

%bitcoin
BTC = readmatrix("BTC GIUSTO.xlsx")
BTCclose = BTC(:,2)
pBTC = log(BTCclose)
yBTC = 100*(diff(pBTC))

%cardano
ADA = readmatrix("ADA.xlsx")
ADAclose = ADA(:,2)
pADA = log(ADAclose)
yADA = 100*(diff(pADA))

%ethereum
ETH = readmatrix("ethereum serie.xlsx")
ETHclose = ETH(:,2)
pETH = log(ETHclose)
yETH = 100*(diff(pETH))

%%%%%%%%%%%%% ADF test %%%%%%%%%%%%%%%%%%%%%

[h1,pValue1,stat1] = adftest (pXRP, Model = 'TS',lags=0:3)   ;
ADF1 = table(pValue1,stat1, 'VariableNames', {'PValue', 'stat'});
[h1,pValue1,stat1,cvalue1,reg] = adftest(pXRP,'model', 'TS' ,Lags=0:3) 
figure('Name','AIC') 
plot(0:3,[reg(1:(3+1)).AIC],'r') 
hold on  
plot(0:3,[reg(1:(3+1)).BIC],'y') 
title('pXRP') 
legend('AIC','BIC') 
hold off 

[h2,pValue2,stat2] = adftest (pLTC, Model = 'TS',lags=0:3)   ;
ADF2 = table(pValue2,stat2, 'VariableNames', {'PValue', 'stat'});
[h2,pValue2,stat2,cvalue2,reg] = adftest(pLTC,'model', 'TS' ,Lags=0:3) 
figure('Name','AIC') 
plot(0:3,[reg(1:(3+1)).AIC],'r') 
hold on  
plot(0:3,[reg(1:(3+1)).BIC],'y') 
title('pLTC') 
legend('AIC','BIC') 
hold off 


[h3,pValue3,stat3] = adftest (pBTC, Model = 'TS',lags=0:3)   ;
ADF3 = table(pValue3,stat3, 'VariableNames', {'PValue', 'stat'});
[h3,pValue3,stat3,cvalue3,reg] = adftest(pBTC,'model', 'TS' ,Lags=0:3)
figure('Name','AIC') 
plot(0:3,[reg(1:(3+1)).AIC],'r') 
hold on  
plot(0:3,[reg(1:(3+1)).BIC],'y') 
title('pBTC') 
legend('AIC','BIC') 
hold off 


[h4,pValue4,stat4] = adftest (pADA, Model = 'TS',lags=0:3)   ;
ADF4 = table(pValue4,stat4, 'VariableNames', {'PValue', 'stat'});
[h4,pValue4,stat4,cvalue4,reg] = adftest(pADA,'model', 'TS' ,Lags=0:3)
figure('Name','AIC') 
plot(0:3,[reg(1:(3+1)).AIC],'r') 
hold on  
plot(0:3,[reg(1:(3+1)).BIC],'y') 
title('pADA') 
legend('AIC','BIC') 
hold off 


[h5,pValue5,stat5] = adftest (pETH, Model = 'TS',lags=0:3)   ;
ADF5 = table(pValue5,stat5, 'VariableNames', {'PValue', 'stat'});
[h5,pValue5,stat5,cvalue5,reg] = adftest(pETH,'model', 'TS' ,Lags=0:3)
figure('Name','AIC') 
plot(0:3,[reg(1:(3+1)).AIC],'r') 
hold on  
plot(0:3,[reg(1:(3+1)).BIC],'y') 
title('pETH') 
legend('AIC','BIC') 
hold off

TESTadf = vertcat(ADF1,ADF2,ADF3,ADF4, ADF5);
TESTadf.Properties.RowNames = {'test-yXRP','test-yLTC','test-yBTC','test-yADA','test-yETH'} ;

%%%%%%%%test di Phillips per le bolle speculative%%%%%%%%

% Applicazione alla serie di XRP
[gsadf_stat, cv] = GSADFtest(pXRP, 40, 250);  

plot(gsadf_stat, 'b', 'LineWidth', 1.5);
hold on;
plot(cv(2)*ones(size(gsadf_stat)), 'r--', 'LineWidth', 1.5);
legend('GSADF Statistic', 'Critical Value (95%)');
xlabel('Time');
ylabel('GSADF Statistic');
title('GSADF Test for Bubbles in XRP');
grid on;
hold off;

startBubbleXRP = date1(30);
endBubbleXRP = date1(65);
disp(['Periodo di bolla XRP: da ', datestr(startBubbleXRP), ' a ', datestr(endBubbleXRP)])


% Applicazione alla serie di BTC
[gsadf_stat, cv] = GSADFtest(pBTC, 40, 250);

plot(gsadf_stat, 'b', 'LineWidth', 1.5);
hold on;
plot(cv(2)*ones(size(gsadf_stat)), 'r--', 'LineWidth', 1.5);
legend('GSADF Statistic', 'Critical Value (95%)');
xlabel('Time');
ylabel('GSADF Statistic');
title('GSADF Test for Bubbles in BTC');
grid on;
hold off;

% Applicazione alla serie di LTC
[gsadf_stat, cv] = GSADFtest(pLTC, 40, 250);

plot(gsadf_stat, 'b', 'LineWidth', 1.5);
hold on;
plot(cv(2)*ones(size(gsadf_stat)), 'r--', 'LineWidth', 1.5);
legend('GSADF Statistic', 'Critical Value (95%)');
xlabel('Time');
ylabel('GSADF Statistic');
title('GSADF Test for Bubbles in LTC');
grid on;
hold off;

% Applicazione alla serie di ADA
[gsadf_stat, cv] = GSADFtest(pADA, 40, 250);

plot(gsadf_stat, 'b', 'LineWidth', 1.5);
hold on;
plot(cv(2)*ones(size(gsadf_stat)), 'r--', 'LineWidth', 1.5);
legend('GSADF Statistic', 'Critical Value (95%)');
xlabel('Time');
ylabel('GSADF Statistic');
title('GSADF Test for Bubbles in ADA');
grid on;
hold off;

startBubbleADA = date4(20);
endBubbleADA = date4(60);
disp(['Periodo di bolla ADA: da ', datestr(startBubbleADA), ' a ', datestr(endBubbleADA)])

% Applicazione alla serie di ETH
[gsadf_stat, cv] = GSADFtest(pETH, 40, 250);

plot(gsadf_stat, 'b', 'LineWidth', 1.5);
hold on;
plot(cv(2)*ones(size(gsadf_stat)), 'r--', 'LineWidth', 1.5);
legend('GSADF Statistic', 'Critical Value (95%)');
xlabel('Time');
ylabel('GSADF Statistic');
title('GSADF Test for Bubbles in ETH');
grid on;
hold off;


%%%%%%%%%%Varianza di Cochrane TEST %%%%%%%%%%%

n = length (yXRP);
[h6,pValue6,stat6] = vratiotest(yXRP, period = round(n^0.33), IID = false);
Coch1 = table(h6,pValue6,stat6, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yLTC);
[h7,pValue7,stat7] = vratiotest(yLTC, period = round(n^0.33), IID = false);
Coch2 = table(h7,pValue7,stat7, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yBTC);
[h8,pValue8,stat8] = vratiotest(yBTC, period = round(n^0.33), IID = false);
Coch3 = table(h8,pValue8,stat8, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yADA);
[h9,pValue9,stat9] = vratiotest(yADA, period = round(n^0.33), IID = false);
Coch4 = table(h9,pValue9,stat9, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yETH);
[h10,pValue10,stat10] = vratiotest(yETH, period = round(n^0.33), IID = false);
Coch5 = table(h10,pValue10,stat10, 'VariableNames', {'h', 'PValue', 'stat'});

% Aggiungi una colonna per identificare ogni criptovaluta
Coch1.Crypto = repmat("XRP", height(Coch1), 1);
Coch2.Crypto = repmat("LTC", height(Coch2), 1);
Coch3.Crypto = repmat("BTC", height(Coch3), 1);
Coch4.Crypto = repmat("ADA", height(Coch4), 1);
Coch5.Crypto = repmat("ETH", height(Coch5), 1);

CochTot = [Coch1; Coch2; Coch3; Coch4; Coch5];
CochTot = CochTot(:, {'Crypto', 'h', 'PValue', 'stat'});
disp(CochTot);


%%%%%%%%%%%% GRAFICI DELLE SERIE %%%%%%%%%%%%%%%%

% PRIMA SERIE XRP

T = readtable("XRP.xlsx", 'VariableNamingRule', 'preserve');
date1 = T.Date; 

figure(1)
% Primo subplot per pXRP
subplot(2,1,1); 
plot(date1,pXRP, 'b', 'LineWidth', 1); 
title('XRP - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

% Secondo subplot per yXRP
subplot(2, 1, 2); 
plot(date1(2:end),yXRP, 'r', 'LineWidth', 1); 
title('XRP - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(2)
% autocorrelazione campionaria globale con 30 ritardi
subplot(2,1,1); autocorr(yXRP,30) 
%autocorrelazione campionaria parziale con 30 ritardi
subplot(2,1,2); parcorr(yXRP,30)

%calcolo della media dei rendimenti logaritmici
mu = mean(yXRP)   ;

%calcolo della mediana dei rendimenti logaritmici
Me = median(yXRP) ;

%calcolo della varianza e deviazione standard dei rendimenti logaritmici
Var = var(yXRP)  ;
Stdev = std(yXRP);

%calcolo  dell'indice di Sharp dei rendimenti logaritmici assumendo Rf=0
Rf = 0  ;
SR = (mu-Rf)/Stdev ;

%calcolo dell'asimmetria della distribuzione
 
A = skewness(yXRP) ;

%calcolo della curtosi
K = kurtosis(yXRP);

%calcolo dello scarto interquantile
IQR = iqr(yXRP) ;

% ISTOGRAMMA DEI RENDIMENTI LOGARITMICI
figure(3)
histfit(yXRP)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yXRP,'Normal') ;

%per verificare ulterioremente la non normalità della distribuzione è stato
%eseguito un test-JB e un qqplot per conforntare i quantili della stessa
%con quelli della distribuzione normale
[HJB,PvalueJB,JBstat,critJB] = jbtest(yXRP) ;
qqplot(yXRP) ;


%SECONDA SERIE LTC
T = readtable("LTC .xlsx", 'VariableNamingRule', 'preserve');
date2 = T.Date; 

figure(4)
subplot(2,1,1); 
plot(date2,pLTC, 'b', 'LineWidth', 1); 
title('LTC - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(date2(2:end),yLTC, 'r', 'LineWidth', 1); 
title('LTC - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(5)
subplot(2,1,1); autocorr(yLTC,30) 
subplot(2,1,2); parcorr(yLTC,30)

mu2 = mean(yLTC)   ;
Me2 = median(yLTC) ;

Var2 = var(yLTC)  ;
Stdev2 = std(yLTC);

Rf = 0  ;
SR2 = (mu2-Rf)/Stdev2 ;
 
A2 = skewness(yLTC) ;
K2 = kurtosis(yLTC);
IQR2 = iqr(yLTC) ;

figure(6)
histfit(yLTC)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yLTC,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yLTC) ;
qqplot(yLTC) ;

%TERZA SERIE BTC
T = readtable("BTC GIUSTO.xlsx", 'VariableNamingRule', 'preserve');
date3 = T.Date; 

figure(7)
subplot(2,1,1); 
plot(date3,pBTC, 'b', 'LineWidth', 1); 
title('BTC - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')


subplot(2, 1, 2); 
plot(date3(2:end),yBTC, 'r', 'LineWidth', 1); 
title('BTC - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(8)
subplot(2,1,1); autocorr(yBTC,90) 
subplot(2,1,2); parcorr(yBTC,90)

mu3 = mean(yBTC)   ;
Me3 = median(yBTC) ;

Var3 = var(yBTC)  ;
Stdev3 = std(yBTC);

Rf = 0  ;
SR3 = (mu3-Rf)/Stdev3 ;
 
A3 = skewness(yBTC) ;
K3 = kurtosis(yBTC);
IQR3 = iqr(yBTC) ;

figure(9)
histfit(yBTC)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yBTC,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yBTC) ;
qqplot(yBTC) ;

%QUARTA SERIE ADA
T = readtable("ADA.xlsx", 'VariableNamingRule', 'preserve');
date4 = T.Date; 

figure(10)
subplot(2,1,1); 
plot(date4,pADA, 'b', 'LineWidth', 1); 
title('ADA - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(date4(2:end),yADA, 'r', 'LineWidth', 1); 
title('ADA - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(11)
subplot(2,1,1); autocorr(yADA,30) 
subplot(2,1,2); parcorr(yADA,30)

mu4 = mean(yADA)   ;
Me4 = median(yADA) ;

Var4 = var(yADA)  ;
Stdev4 = std(yADA);

Rf = 0  ;
SR4 = (mu4-Rf)/Stdev4 ;
 
A4 = skewness(yADA) ;
K4 = kurtosis(yADA);
IQR4 = iqr(yADA) ;

figure(12)
histfit(yADA)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yADA,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yADA) ;
qqplot(yADA) ;

%QUINTA SERIE ETH
T = readtable("ethereum serie.xlsx", 'VariableNamingRule', 'preserve');
date5 = T.Date; 

figure(13)
subplot(2,1,1); 
plot(date5,pETH, 'b', 'LineWidth', 1); 
title('ETH - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(date5(2:end),yETH, 'r', 'LineWidth', 1); 
title('ETH - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(14)
subplot(2,1,1); autocorr(yETH,30) 
subplot(2,1,2); parcorr(yETH,30)

mu5 = mean(yETH)   ;
Me5 = median(yETH) ;

Var5 = var(yETH)  ;
Stdev5 = std(yETH);

Rf = 0  ;
SR5 = (mu5-Rf)/Stdev5 ;
 
A5 = skewness(yETH) ;
K5 = kurtosis(yETH);
IQR5 = iqr(yETH) ;

figure(15)
histfit(yETH)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yETH,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yETH) ;
qqplot(yETH) ;

%%%%%%%%%%%%%%%%%%%% STIMA DENSITA SPETTARLE %%%%%%%%%%%%%%%

% STIMA DELLA DENSITA' SPETTRALE XRP
% Calcolo dei rendimenti semplici
yXRP3 = diff(XRPclose) ./ XRPclose(1:end-1);  % rendimenti semplici


% Stima densità spettrale
omega = 0:0.001:pi; 
n = length(yXRP3);
m = round(n^(1/3));
C = m / n;

Sdf = fBartlettSpectralDensityEst(yXRP3, m, omega);

figure(17)
plot(omega, Sdf, 'LineWidth', 2)
xlim([0 pi])
title('Stima della densità spettrale di XRP (rendimenti semplici)')
xlabel('\omega')
ylabel('Spettro')

% STIMA DELLA DENSITA' SPETTRALE LTC
yLTC3 = diff(LTCclose) ./ LTCclose(1:end-1);
figure(17)
plot(date2(2:end),yLTC3)
omega = 0:0.001:pi; 
length (omega) 
n = length (yLTC3);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yLTC3, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di LTC')

% STIMA DELLA DENSITA' SPETTRALE BTC
yBTC3 = diff(BTCclose) ./ BTCclose(1:end-1);
figure(18)
plot(date3(2:end),yBTC3)
omega = 0:0.001:pi; 
length (omega) 
n = length (yBTC3);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yBTC3, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di BTC')

% STIMA DELLA DENSITA' SPETTRALE ADA
yADA3 = diff(ADAclose) ./ ADAclose(1:end-1);
figure(19)
plot(date4(2:end),yADA3)
omega = 0:0.001:pi; 
length (omega) 
n = length (yADA3);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yADA3, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di ADA')

% STIMA DELLA DENSITA' SPETTRALE ETH
yETH3 = diff(ETHclose) ./ ETHclose(1:end-1);
figure(20)
plot(date5(2:end),yETH3)
omega = 0:0.001:pi; 
length (omega) 
n = length (yETH3);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yETH3, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di ETH')


%%%%%%%%%%%%%  STIMA DEI MODELLI %%%%%%%%%% 
%%%%%%%%%%%%%%%%% GARCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%
size(yXRP)
size(yLTC)
size(yBTC)
size(yADA)
size(yETH)

% 1. Assicurati che tutte le serie abbiano la stessa lunghezza
minLength = min([length(yXRP), length(yLTC), length(yBTC), length(yADA), length(yETH)]);
yXRP = yXRP(1:minLength);
yLTC = yLTC(1:minLength);
yBTC = yBTC(1:minLength);
yADA = yADA(1:minLength);
yETH = yETH(1:minLength);
mY = [yXRP, yLTC, yBTC, yADA, yETH];

% 3. Rimuovi eventuali NaN
mY = mY(~any(isnan(mY),2), :);


model = garch(1,1);
garchModels = cell(1,5);
for i = 1:5
    garchModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri GARCH stimati:')
for i = 1:5
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f\n', i, ...
        garchModels{i}.Constant, garchModels{i}.ARCH{1}, garchModels{i}.GARCH{1});
end

% 7. Previsione della volatilità a 10 giorni
forecastHorizon = 10;
volatilityForecast = zeros(forecastHorizon, 5);

for i = 1:5
    v = forecast(garchModels{i}, forecastHorizon);
    volatilityForecast(:, i) = sqrt(v); % Volatilità = radice della varianza prevista
end

% 8. Visualizza la previsione della volatilità
disp('Previsione della volatilità a 10 giorni per ciascuna serie:')
disp(volatilityForecast)


%%%%%%%%%%%%%%%% modello e garch %%%%%%%%%%%%%%%%%%%%
varianza = var(mY);
disp('Varianza di ciascuna serie:')
disp(varianza)
model = egarch(1,1);
egarchModels = cell(1,5);

for i = 1:5
    egarchModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri EGARCH stimati:')
for i = 1:5
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f, Gamma = %.5f\n', i, ...
        egarchModels{i}.Constant, egarchModels{i}.ARCH{1}, ...
        egarchModels{i}.GARCH{1}, egarchModels{i}.Leverage{1});
end

% 6. Previsione della volatilità a 10 giorni
forecastHorizon = 10;
volatilityForecast = zeros(forecastHorizon, 5);

for i = 1:5
    v = forecast(egarchModels{i}, forecastHorizon);
    volatilityForecast(:, i) = sqrt(v); % Volatilità = radice della varianza prevista
end

% 7. Visualizza la previsione della volatilità
disp('Previsione della volatilità a 10 giorni per ciascuna serie:')
disp(volatilityForecast)


%%%%%%%%%%%% modello GJR-GARCH %%%%%%%%%%%%%%%%%%

model = gjr(1,1);  
gjrModels = cell(1,5);

for i = 1:5
    gjrModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri GJR-GARCH stimati:')
for i = 1:5
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f, Gamma = %.5f\n', i, ...
        gjrModels{i}.Constant, gjrModels{i}.ARCH{1}, ...
        gjrModels{i}.GARCH{1}, gjrModels{i}.Leverage{1});
end

% 5. Previsione della volatilità a 10 giorni
forecastHorizon = 10;
volatilityForecast = zeros(forecastHorizon, 5);

for i = 1:5
    v = forecast(gjrModels{i}, forecastHorizon);
    volatilityForecast(:, i) = sqrt(v); % Volatilità = radice della varianza prevista
end

% 6. Visualizza la previsione della volatilità
disp('Previsione della volatilità a 10 giorni per ciascuna serie:')
disp(volatilityForecast)

%%%%%%%%%%% MODELLO T-GAS %%%%%%%%%%%%%%%%%%%%%

% Numero di serie storiche
numSeries = 5;

% Inizializza i parametri
omega = zeros(1, numSeries);
alpha = zeros(1, numSeries);
beta = zeros(1, numSeries);

% Numero di iterazioni per ottimizzazione
maxIter = 1000;

% Loop sulle serie storiche
for i = 1:numSeries
    % Standardizzazione della serie storica
    y = (mY(:, i) - mean(mY(:, i))) / std(mY(:, i));
    lagY = [NaN; y(1:end-1)]; % Aggiunge un ritardo alla serie

    % Definizione della funzione di log-verosimiglianza
    logLikFunc = @(params) -sum(log(tpdf(abs(y(2:end) ./ exp(params(1) + params(2) * y(2:end) + params(3) * lagY(2:end))), 5)));

    % Definizione dei parametri iniziali e dei vincoli
    initParams = [log(var(y) + eps), 0.2, 0.5]; % Aggiunto eps per evitare log(0)
    lb = [-Inf, 0.01, 0.01]; % Vincoli: alpha e beta >= 0
    ub = [Inf, 1, 1];  % Vincoli: alpha e beta <= 1

    % Impostazioni dell'ottimizzazione
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'MaxIterations', maxIter, 'Display', 'iter');

    % Ottimizzazione con fmincon
    estimatedParams = fmincon(logLikFunc, initParams, [], [], [], [], lb, ub, [], options);

    % Salvataggio dei parametri stimati
    omega(i) = estimatedParams(1);
    alpha(i) = estimatedParams(2);
    beta(i) = estimatedParams(3);
end

% Stampa i parametri stimati
disp('Parametri stimati per il modello t-GAS:')
for i = 1:numSeries
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f\n', i, omega(i), alpha(i), beta(i));
end


%%%%%%%%%%%%%%%%%%% MODELLO LSTM %%%%%%%%%%%%%%%%%%%%%%%

% Normalizza le serie storiche per migliorare la stabilità del training
numSeries = 5;
numTimesteps = size(mY,1); % Numero di osservazioni nel tempo

mY2 = zeros(size(mY));

for i = 1:numSeries
    mY2(:,i) = (mY(:,i) - mean(mY(:,i))) / std(mY(:,i)); % Standardizzazione
end

% Definisce il numero di lag (finestra di osservazione)
sequenceLength = 10; % Usa gli ultimi 10 giorni per prevedere il successivo

XTrain = {};
YTrain = {};

for i = 1:numSeries
    for t = 1:(numTimesteps-sequenceLength)
        XTrain{end+1} = mY2(t:t+sequenceLength-1, i)'; % Input sequence (trasposta per adattarsi a LSTM)
        YTrain{end+1} = mY2(t+sequenceLength, i); % Target (previsione del prossimo valore)
    end
end

% Definizione dell'architettura LSTM
layers = [
    sequenceInputLayer(10)  % Una feature per timestep
    lstmLayer(50, 'OutputMode', 'last')  
    fullyConnectedLayer(1)  
    regressionLayer 
];

% Opzioni di training
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress'); 

% Concatena le sequenze lungo la dimensione corretta
XTrain = cat(1, XTrain{:}); % Ora XTrain dovrebbe essere [numCampioni, sequenceLength]

% Trasponi per ottenere la forma [sequenceLength, numCampioni]
XTrain = XTrain'; % Ora è [10, 555]

% Converti in cell array, ogni cella contiene una sequenza [10 × 1]
XTrain = num2cell(XTrain, 1)'; % Ora è [555, 1]
size(XTrain)

% Converti YTrain in cell array
YTrain = cell2mat(YTrain); % Converte la cell array in un array numerico
YTrain = YTrain(:); % Assicura che sia un vettore colonna
size(YTrain)

% Allena la rete LSTM
net = trainNetwork(XTrain, YTrain, layers, options);

% Previsione sui prossimi 10 giorni per ciascuna serie
forecastHorizon = 10;
predictions = zeros(forecastHorizon, numSeries);

for i = 1:numSeries
    lastSequence = mY2(end-sequenceLength+1:end, i); % Ora è [10 × 1]
    
    for t = 1:forecastHorizon
        pred = predict(net, lastSequence); % Predizione LSTM
        predictions(t, i) = pred;
        
        % Aggiorna la sequenza con la nuova previsione
        lastSequence = [lastSequence(2:end); pred]; % Mantiene la forma [10 × 1]
    end
end

% Visualizza le previsioni
disp('Previsioni LSTM per i prossimi 10 giorni:')
disp(predictions)


%%%%%%%%% richiesta professore %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%2. Calcola rendimenti logaritmici normalizzati %%%%%%%%%%%
numSeries = size(mY,2);
numTimesteps = size(mY,1);
sequenceLength = 10;
forecastHorizon = 10;

% Normalizza le serie (stessa procedura del training!)
means = mean(mY(1:end-forecastHorizon,:));
stds  = std(mY(1:end-forecastHorizon,:));
mY2 = (mY - means) ./ stds;

% === 3. Estrai i rendimenti reali dei 10 giorni successivi ===
realReturns = mY2(end-forecastHorizon+1:end, :); % ultimi 10 giorni normalizzati

% === 4. Usa le previsioni già calcolate (predictions) ===
% Assicurati che 'predictions' sia già presente nel workspace

% === 5. Calcola gli errori MSE e MAE ===
mseVals = mean((predictions - realReturns).^2);
maeVals = mean(abs(predictions - realReturns));

% === 6. Stampa risultati ===
cryptoNames = {'XRP','LTC','BTC','ADA','ETH'};
fprintf('\nValutazione performance LSTM (su scala normalizzata):\n');
fprintf('--------------------------------------------------------\n');
fprintf('%6s\t\tMSE\t\t\tMAE\n', 'Crypto');
fprintf('--------------------------------------------------------\n');
for i = 1:numSeries
    fprintf('%6s\t\t%.5f\t\t%.5f\n', cryptoNames{i}, mseVals(i), maeVals(i));
end


clc; clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%VALUTAZIONE FONDI ETF %%%%%%%%%%%%%%%%%%%%%%%%%
ETFAMP = readmatrix("ETF AMPLIFY.xlsx")
ETFAMPclose = ETFAMP(10:end,2)
pETFAMP = log(ETFAMPclose)
yETFAMP = 100*(diff(pETFAMP))

ETFbit = readmatrix("ETF BITWISE.xlsx")
ETFbitclose = ETFbit(10:end,2)
pETFbit = log(ETFbitclose)
yETFbit = 100*(diff(pETFbit))

ETF500 = readmatrix("S&P500.xlsx")
ETF500close = ETF500(10:end,2)
pETF500 = log(ETF500close)
yETF500 = 100*(diff(pETF500))

%%%%%%%%%%%% GRAFICI DELLE SERIE %%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%% GRAFICO ETF AMP %%%%%%%%%%%%%%%%%%%%
ETFAMPtable = readtable("ETF AMPLIFY.xlsx", 'VariableNamingRule', 'preserve');
dateAMP = ETFAMPtable.Price(9:end);

figure(20)

subplot(2,1,1); 
plot(dateAMP,pETFAMP, 'b', 'LineWidth', 1); 
title('ETF AMP - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(dateAMP(2:end),yETFAMP, 'r', 'LineWidth', 1); 
title('ETF AMP - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(21)
subplot(2,1,1); autocorr(yETFAMP,30) 
subplot(2,1,2); parcorr(yETFAMP,30)

mu1 = mean(yETFAMP)   ;
Me1 = median(yETFAMP) ;
Var1 = var(yETFAMP)  ;
Stdev1 = std(yETFAMP);
Rf = 0  ;
SR1 = (mu1-Rf)/Stdev1 ;
A1 = skewness(yETFAMP) ;
K1 = kurtosis(yETFAMP);
IQR1 = iqr(yETFAMP) ;

% ISTOGRAMMA DEI RENDIMENTI LOGARITMICI
figure(22)
histfit(yETFAMP)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yETFAMP,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yETFAMP) ;
qqplot(yETFAMP) ;


%%%%%%%%%%%%% GRAFICO SERIE ETF BITWISE %%%%%%%%%%%%%%%%%%%%%%%%
ETFbitwisetable = readtable("ETF BITWISE.xlsx", 'VariableNamingRule', 'preserve');
datebit = ETFbitwisetable.Price(9:end);

figure(23)
subplot(2,1,1); 
plot(datebit,pETFbit, 'b', 'LineWidth', 1); 
title('ETF BITWISE - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(datebit(2:end),yETFbit, 'r', 'LineWidth', 1); 
title('ETF BITWISE - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(24)
subplot(2,1,1); autocorr(yETFbit,30) 
subplot(2,1,2); parcorr(yETFbit,30)
mu2 = mean(yETFbit)   ;
Me2 = median(yETFbit) ;
Var2 = var(yETFbit)  ;
Stdev2 = std(yETFbit);
Rf = 0  ;
SR2 = (mu2-Rf)/Stdev2 ;
A2 = skewness(yETFbit) ;
K2 = kurtosis(yETFbit);
IQR2 = iqr(yETFbit) ;

% ISTOGRAMMA DEI RENDIMENTI LOGARITMICI
figure(25)
histfit(yETFbit)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yETFbit,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yETFbit) ;
qqplot(yETFbit) ;

%%%%%%%%%%%%% GRAFICO SERIE ETF S&P500 %%%%%%%%%%%%%%%%%%%%%%%%
ETF500 = readtable("S&P500.xlsx", 'VariableNamingRule', 'preserve');
date500 = ETF500.Price(9:end);

figure(26)
subplot(2,1,1); 
plot(date500,pETF500, 'b', 'LineWidth', 1); 
title('ETF S&P500 - Prezzi di chiusura log');
xlabel('tempo')
ylabel('prezzi logaritmici')

subplot(2, 1, 2); 
plot(date500(2:end),yETF500, 'r', 'LineWidth', 1); 
title('ETF S&P500 - Rendimenti logartimici');
xlabel('tempo')
ylabel('rendimenti logaritmici')

figure(27)
subplot(2,1,1); autocorr(yETF500,30) 
subplot(2,1,2); parcorr(yETF500,30)

mu3 = mean(yETF500)   ;
Me3 = median(yETF500) ;
Var3 = var(yETF500)  ;
Stdev3 = std(yETF500);
Rf = 0  ;
SR3 = (mu3-Rf)/Stdev3 ;
A3 = skewness(yETF500) ;
K3 = kurtosis(yETF500);
IQR3 = iqr(yETF500) ;

% ISTOGRAMMA DEI RENDIMENTI LOGARITMICI
figure(28)
histfit(yETF500)
title('ISTOGRAMMA DEI RENDIMENTI LOGARITMICI vs NORMALE')
pardistr = fitdist(yETF500,'Normal') ;

[HJB,PvalueJB,JBstat,critJB] = jbtest(yETF500) ;
qqplot(yETF500) ;

%%%%%%%%%%%%% TEST ADF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[h1,pValue1,stat1] = adftest (pETFAMP, Model = 'TS',lags=1:3)   ;
ADF1 = table(h1,pValue1,stat1, 'VariableNames', {'h', 'PValue', 'stat'});

[h2,pValue2,stat2] = adftest (pETFbit, Model = 'TS',lags=1:3)   ;
ADF2 = table(h2,pValue2,stat2, 'VariableNames', {'h', 'PValue', 'stat'});

[h3,pValue3,stat3] = adftest (pETF500, Model = 'TS',lags=1:3)   ;
ADF3 = table(h3,pValue3,stat3, 'VariableNames', {'h', 'PValue', 'stat'});

TESTadf = vertcat(ADF1,ADF2,ADF3);
TESTadf.Properties.RowNames = {'test-yETFAMP','test-ETFbit','test-RTF500'} ;

%%%%%%%%%%%%%%%%%%%%%% Varianza di Cochrane TEST %%%%%%%%%%%%%%%%%%%
n = length (yETFAMP);
[h4,pValue4,stat4] = vratiotest(yETFAMP, period = round(n^0.33), IID = false);
Coch1 = table(h4,pValue4,stat4, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yETFbit);
[h5,pValue5,stat5] = vratiotest(yETFbit, period = round(n^0.33), IID = false);
Coch2 = table(h5,pValue5,stat5, 'VariableNames', {'h', 'PValue', 'stat'});

n = length (yETF500);
[h6,pValue6,stat6] = vratiotest(yETF500, period = round(n^0.33), IID = false);
Coch3 = table(h6,pValue6,stat6, 'VariableNames', {'h', 'PValue', 'stat'});

Coch1.ETF = repmat("AMP", height(Coch1), 1);
Coch2.ETF = repmat("bit", height(Coch2), 1);
Coch3.ETF = repmat("500", height(Coch3), 1);

CochTot = [Coch1; Coch2; Coch3];
CochTot = CochTot(:, {'ETF', 'h', 'PValue', 'stat'});
disp(CochTot);

%%%%%%%%%%%%%%%%%%%% STIMA DENSITA SPETTARLE %%%%%%%%%%%%%%%

% STIMA DELLA DENSITA' SPETTRALE ETFAMPLIFY
yETFAMP2 = yETFAMP .^ 2 ;
figure(29)
plot(dateAMP(2:end),yETFAMP2)
omega = 0:0.001:pi; 
length (omega) 
n = length (yETFAMP);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yETFAMP2, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di ETF AMPLIFY')

% STIMA DELLA DENSITA' SPETTRALE ETFBITWISE
yETFbit2 = yETFbit .^ 2 ;
figure(30)
plot(datebit(2:end),yETFbit2)
omega = 0:0.001:pi; 
length (omega) 
n = length (yETFbit);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yETFbit2, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di ETF BITWISE')

% STIMA DELLA DENSITA' SPETTRALE ETFS&P500
yETF5002 = yETF500 .^ 2 ;
figure(29)
plot(date500(2:end),yETF5002)
omega = 0:0.001:pi; 
length (omega) 
n = length (yETF500);
m = round(n^(1/3));
C  = m/n; 
Sdf = fBartlettSpectralDensityEst(yETF5002, m, omega) ;
plot(omega,Sdf,'LineWidth',2)
xlim([0 pi]) 
title('Stima della densità spettrale di ETF S&P500')

%%%%%%%%%%%%%%%%%%%  STIMA DEI MODELLI %%%%%%%%%%%%%%%%%%%%

size(yETFAMP)
size(yETFbit)
size(yETF500)

% 1. Assicurati che tutte le serie abbiano la stessa lunghezza
minLength = min([length(yETFAMP), length(yETFbit), length(yETF500)]);
yETFAMP = yETFAMP(1:minLength);
yETFbit = yETFbit(1:minLength);
yETF500 = yETF500(1:minLength);

mY = [yETFAMP, yETFbit, yETF500];

mY = mY(~any(isnan(mY),2), :);

%%%%%%%%%%%%%%%%% GARCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = garch(1,1);
garchModels = cell(1,3);
for i = 1:3
    garchModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri GARCH stimati:')
for i = 1:3
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f\n', i, ...
        garchModels{i}.Constant, garchModels{i}.ARCH{1}, garchModels{i}.GARCH{1});
end


%%%%%%%%%%%%%%%% modello e garch %%%%%%%%%%%%%%%%%%%%
varianza = var(mY);
disp('Varianza di ciascuna serie:')
disp(varianza)
model = egarch(1,1);
egarchModels = cell(1,3);

for i = 1:3
    egarchModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri EGARCH stimati:')
for i = 1:3
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f, Gamma = %.5f\n', i, ...
        egarchModels{i}.Constant, egarchModels{i}.ARCH{1}, ...
        egarchModels{i}.GARCH{1}, egarchModels{i}.Leverage{1});
end

%%%%%%%%%%%% modello GJR-GARCH %%%%%%%%%%%%%%%%%%
model = gjr(1,1);  
gjrModels = cell(1,3);

for i = 1:3
    gjrModels{i} = estimate(model, mY(:,i)); 
end

disp('Parametri GJR-GARCH stimati:')
for i = 1:3
    fprintf('Serie %d: Omega = %.5f, Alpha = %.5f, Beta = %.5f, Gamma = %.5f\n', i, ...
        gjrModels{i}.Constant, gjrModels{i}.ARCH{1}, ...
        gjrModels{i}.GARCH{1}, gjrModels{i}.Leverage{1});
end

%%%%%%%%%%%%%%%%%%% MODELLO LSTM %%%%%%%%%%%%%%%%%%%%%%%
numSeries = 3;
numTimesteps = size(mY,1); 

mY2 = zeros(size(mY));

for i = 1:numSeries
    mY2(:,i) = (mY(:,i) - mean(mY(:,i))) / std(mY(:,i)); 
end

sequenceLength = 10; 

XTrain = {};
YTrain = {};

for i = 1:numSeries
    for t = 1:(numTimesteps-sequenceLength)
        XTrain{end+1} = mY2(t:t+sequenceLength-1, i)'; 
        YTrain{end+1} = mY2(t+sequenceLength, i); 
    end
end

layers = [
    sequenceInputLayer(10)  
    lstmLayer(50, 'OutputMode', 'last')  
    fullyConnectedLayer(1)  
    regressionLayer 
];


options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress'); 

XTrain = cat(1, XTrain{:}); 
XTrain = XTrain'; 
XTrain = num2cell(XTrain, 1)'; 
size(XTrain)
YTrain = cell2mat(YTrain); 
YTrain = YTrain(:); 
size(YTrain)
net = trainNetwork(XTrain, YTrain, layers, options);

%%%%%%%%%%% Parametri Processo di Wiener e Moto Browniano Geometrico
%%%%%%%%%%% (GBM)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu4 = mu1/100; % Drift stimato (media dei rendimenti)
sigma = Stdev1 / 100; % Volatilità stimata
S0 = ETFAMPclose(end);      % Ultimo prezzo osservato come valore iniziale
T = 1;                      % Orizzonte temporale (1 anno)
dt = 1/252;                 % Passo temporale (giornaliero)
N = T/dt;                   % Numero di passi
nPaths = 1000;              % Numero di simulazioni

% Definizione del modello GBM
gbmModel = gbm(mu4, sigma, 'StartState', S0);

% Simulazione dei prezzi
[Paths, Times] = simulate(gbmModel, N, 'DeltaTime', dt, 'nTrials', nPaths);

% Grafico delle simulazioni
figure;
plot(Times, squeeze(Paths(:,1:10)));
title('Simulazioni GBM per l’ETF AMPLIFY');
xlabel('Tempo');
ylabel('Prezzo simulato');


%%%%%%%%%%%%%%%%%%%%%      DCC  tra criptovalute  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

minLength = min([length(yXRP), length(yLTC), length(yBTC), length(yADA), length(yETH)]);
yXRP = yXRP(1:minLength);
yLTC = yLTC(1:minLength);
yBTC = yBTC(1:minLength);
yADA = yADA(1:minLength);
yETH = yETH(1:minLength); 

data = table(yXRP,yLTC,yBTC,yADA,yETH,'VariableNames', {'yXRP', 'yLTC', 'yBTC','yADA','yETH'});
disp(data);
my = data;
cN = 5;
vt = date4(2:end) ;
[cn, cN] = size(my);    
my_star = [];  
mcond_stdev = []; 
for i=1:cN
    [coeff, stds_residuals, cond_stdev] = fgarch11_fit(my{:, i}) ;
    my_star = [my_star  stds_residuals];
    mcond_stdev = [mcond_stdev  cond_stdev];
end
figure('Name','Conditional standard deviations \sqrt{h_{ii,t}}');
subplot(2,1,1); plot(vt, my_star); title('Rendimenti Standardizzati');
subplot(2,1,2); plot(vt, mcond_stdev); title('Deviazione Standard Condizionata');



%%%%%%%%%%%%%%%%%% MLE of a and b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting values for parameters 
da =  (0.05); db=  (0.8);
mQbar = cov(my_star,1);
vPsi0 = [log(da/(1-da)); log(db/(1-db)) ];
%
f  = @(vPsi)fDCC_LogLikelihood(my_star, mQbar, vPsi); % function 
opts = optimset('Display','iter','TolX',1e-4,'TolFun',1e-4,...
                'Diagnostics','off', 'MaxIter',1000, 'MaxFunEvals', 1000,...
                'LargeScale', 'off', 'PlotFcns', @optimplotfval);
[vPsi, fval, exitflag, output] = fminunc(f, vPsi0, opts);
disp('Estimation results');
[mloglik, da, db] = fDCC_LogLikelihood(my_star, mQbar, vPsi);

disp(['Parameter a: ', num2str(da)]);
disp(['Parameter b: ', num2str(db)]);


%% Estimation Results
disp(['Parameter a: ', num2str(da)]);
disp(['Parameter b: ', num2str(db)]);
disp(['Matrix Qbar: ' ]);  mQbar
aP  = fDCC_cond_corr(my_star, mQbar, da, db);
figure('Name','Cond correlations')
for i=1:cN
    for j=i+1:cN
        acorr = aP(i,j,:);
        plot(vt, acorr(:)); hold on;
    end
end
hold off
title('Correlazioni dinamiche tra criptovalute')


%%%%%%%%%%%%%%  volatilità condizionata e co-volatilità DCC  %%%%%%%%%%%%
aD=NaN(cN,cN,cn)
for i=1:cn
    aD(:,:,i)=diag(mcond_stdev(i,:));
end

Dt=aD(:,:,1520)
aH=pagemtimes(pagemtimes(aD,aP),aD);

figure(44) 
subplot(2,1,1)
for i=1:cN
    for j=i+1:cN
        covarcond = aH(i,j,:);
        plot(vt, covarcond(:)); hold on;
    end
end
hold off
title('Covarianze dinamiche condizionate tra criptovalute')

subplot(2,1,2)
for i=1:cN
        varcond = aH(i,i,:);
        plot(vt, varcond(:)); hold on;
end
hold off
title('Varianze dinamiche condizionate tra criptovalute')

%%%%%%%%%%%%%%%%%%%%%      DCC  tra etf  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

minLength = min([length(yETF500), length(yETFbit), length(yETFAMP)]);
yETF500 = yETF500(1:minLength);
yETFbit = yETFbit(1:minLength);
yETFAMP = yETFAMP(1:minLength);

%nanETF500 = isnan(yETF500);
%nanETFbit = isnan(yETFbit);
%nanETFAMP = isnan(yETFAMP);
%disp('Posizioni NaN in yETF500:'); disp(find(nanETF500));
%disp('Posizioni NaN in yETFbit:'); disp(find(nanETFbit));
%disp('Posizioni NaN in yETFAMP:'); disp(find(nanETFAMP));


% Creiamo una maschera logica per identificare le righe senza NaN
validRows = ~isnan(yETF500) & ~isnan(yETFbit) & ~isnan(yETFAMP);

% Manteniamo solo le righe valide
yETF500 = yETF500(validRows);
yETFbit = yETFbit(validRows);
yETFAMP = yETFAMP(validRows);

data = table(yETF500,yETFbit,yETFAMP,'VariableNames', {'yETF500', 'yETFbit', 'yETFAMP'});
disp(data);
my = data;
cN = 3;
vt = datebit(2:end) ;
vt = vt(validRows); 
[cn, cN] = size(my);    
my_star = [];  
mcond_stdev = []; 

for i=1:cN
    [coeff, stds_residuals, cond_stdev] = fgarch11_fit(my{:, i}) ;
    my_star = [my_star  stds_residuals];
    mcond_stdev = [mcond_stdev  cond_stdev];
end
figure('Name','Conditional standard deviations \sqrt{h_{ii,t}}');
subplot(2,1,1); plot(vt, my_star); title('Rendimenti Standardizzati');
subplot(2,1,2); plot(vt, mcond_stdev); title('Deviazione Standard Condizionata');

%%%%%%%%%%%%%%%%%% MLE of a and b %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting values for parameters 
da =  (0.05); db=  (0.8);
mQbar = cov(my_star,1);
vPsi0 = [log(da/(1-da)); log(db/(1-db)) ];
%
f  = @(vPsi)fDCC_LogLikelihood(my_star, mQbar, vPsi); % function 
opts = optimset('Display','iter','TolX',1e-4,'TolFun',1e-4,...
                'Diagnostics','off', 'MaxIter',1000, 'MaxFunEvals', 1000,...
                'LargeScale', 'off', 'PlotFcns', @optimplotfval);
[vPsi, fval, exitflag, output] = fminunc(f, vPsi0, opts);
disp('Estimation results');
[mloglik, da, db] = fDCC_LogLikelihood(my_star, mQbar, vPsi);

disp(['Parameter a: ', num2str(da)]);
disp(['Parameter b: ', num2str(db)]);


%% Estimation Results
disp(['Parameter a: ', num2str(da)]);
disp(['Parameter b: ', num2str(db)]);
disp(['Matrix Qbar: ' ]);  mQbar
aP  = fDCC_cond_corr(my_star, mQbar, da, db);
figure('Name','Cond correlations')
for i=1:cN
    for j=i+1:cN
        acorr = aP(i,j,:);
        plot(vt, acorr(:)); hold on;
    end
end
hold off
title('Correlazioni dinamiche tra ETF')


%%%%%%%%%%%%%%  volatilità condizionata e co-volatilità DCC  %%%%%%%%%%%%
aD=NaN(cN,cN,cn)
for i=1:cn
    aD(:,:,i)=diag(mcond_stdev(i,:));
end

Dt=aD(:,:,914)
aH=pagemtimes(pagemtimes(aD,aP),aD);

figure(45) 
subplot(2,1,1)
for i=1:cN
    for j=i+1:cN
        covarcond = aH(i,j,:);
        plot(vt, covarcond(:)); hold on;
    end
end
hold off
title('Covarianze dinamiche condizionate tra ETF')

subplot(2,1,2)
for i=1:cN
        varcond = aH(i,i,:);
        plot(vt, varcond(:)); hold on;
end
hold off
title('Varianze dinamiche condizionate tra ETF')


%%%%%%%%%%%%%%dcc tra etf e crypto %%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% DCC tra criptovalute ed ETF (8 serie) %%%%%%%%%%%%%%%%%%%%%

% Uniforma la lunghezza delle serie
minLength = min([length(yXRP), length(yLTC), length(yBTC), length(yADA), ...
                 length(yETH), length(yETF500), length(yETFbit), length(yETFAMP)]);

% Troncamento delle serie
yXRP = yXRP(1:minLength);
yLTC = yLTC(1:minLength);
yBTC = yBTC(1:minLength);
yADA = yADA(1:minLength);
yETH = yETH(1:minLength);
yETF500 = yETF500(1:minLength);
yETFbit = yETFbit(1:minLength);
yETFAMP = yETFAMP(1:minLength);

% Rimuovi righe con NaN (se presenti)
validRows = ~isnan(yXRP) & ~isnan(yLTC) & ~isnan(yBTC) & ~isnan(yADA) & ~isnan(yETH) & ...
            ~isnan(yETF500) & ~isnan(yETFbit) & ~isnan(yETFAMP);

% Applica la maschera
yXRP = yXRP(validRows);
yLTC = yLTC(validRows);
yBTC = yBTC(validRows);
yADA = yADA(validRows);
yETH = yETH(validRows);
yETF500 = yETF500(validRows);
yETFbit = yETFbit(validRows);
yETFAMP = yETFAMP(validRows);
dateALL = date4(1:minLength);  % oppure usa datebit, se è più coerente
vt = dateALL; % Niente (2:end), la lunghezza è già corretta dopo validRows
vt = vt(validRows);

%%%vt = dateALL(2:end); % oppure adattalo a qualsiasi variabile temporale usi
%%%vt = vt(validRows);

% Costruzione tabella
data = table(yXRP,yLTC,yBTC,yADA,yETH,yETF500,yETFbit,yETFAMP, ...
    'VariableNames', {'XRP','LTC','BTC','ADA','ETH','ETF500','ETFbit','ETFAMP'});
my = data;
[cn, cN] = size(my); 

% GARCH fitting
my_star = [];  
mcond_stdev = []; 
for i = 1:cN
    [coeff, stds_residuals, cond_stdev] = fgarch11_fit(my{:, i});
    my_star = [my_star  stds_residuals];
    mcond_stdev = [mcond_stdev  cond_stdev];
end

% Plot returns & dev st condizionata
figure;
subplot(2,1,1); plot(vt, my_star); title('Standardized returns (Crypto + ETF)');
subplot(2,1,2); plot(vt, mcond_stdev); title('Conditional Std Dev (Crypto + ETF)');

%%%%%%%%%%%%%%%%%% Stima dei parametri DCC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
da = 0.05; db = 0.8;
mQbar = cov(my_star,1);
vPsi0 = [log(da/(1-da)); log(db/(1-db))];

f  = @(vPsi)fDCC_LogLikelihood(my_star, mQbar, vPsi);
opts = optimset('Display','iter','TolX',1e-4,'TolFun',1e-4,...
                'MaxIter',1000,'MaxFunEvals',1000,'LargeScale','off');

[vPsi, fval, exitflag, output] = fminunc(f, vPsi0, opts);
[~, da, db] = fDCC_LogLikelihood(my_star, mQbar, vPsi);
disp(['a = ', num2str(da), '  b = ', num2str(db)]);

%%%%%%%%%%%%%%%%%%%% Correlazioni dinamiche %%%%%%%%%%%%%%%%%%%%%
aP = fDCC_cond_corr(my_star, mQbar, da, db);
figure;
for i = 1:cN
    for j = i+1:cN
        plot(vt, squeeze(aP(i,j,:))); hold on;
    end
end
title('Correlazioni dinamiche DCC (Crypto + ETF)');
legend;

%%%%%%%%%%%%%%%%%%%% Varianza e Covarianza %%%%%%%%%%%%%%%%%%%%%
aD = NaN(cN,cN,cn);
for i = 1:cn
    aD(:,:,i) = diag(mcond_stdev(i,:));
end
aH = pagemtimes(pagemtimes(aD,aP),aD);

figure;
subplot(2,1,1)
for i = 1:cN
    for j = i+1:cN
        plot(vt, squeeze(aH(i,j,:))); hold on;
    end
end
title('Covarianze condizionate DCC (Crypto + ETF)');

subplot(2,1,2)
for i = 1:cN
    plot(vt, squeeze(aH(i,i,:))); hold on;
end
title('Varianze condizionate DCC (Crypto + ETF)');








