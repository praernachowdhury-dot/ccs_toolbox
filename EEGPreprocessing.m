eeglab nogui

inputDir        = uigetdir(pwd,'Select base Directory');
outputDir       = uigetdir(pwd,'Select Output Directory');
logfilename     = sprintf('PreprocessingPart01Log_%s.csv',datetime('now','Format','yyyyMMdd_HHmm'));

filelist    = dir(fullfile(inputDir,'*_EEG.set'));

preprocessedfiles = dir(fullfile(outputDir,'*_ICA.set'));
preprocessedfiles = {preprocessedfiles.name}';

for file_no = 1:length(filelist)
    try    
        eegfilepath     = filelist(file_no).folder; 
        filename        = filelist(file_no).name;

        %% This step required if want to pre-process for some specific files
        % if any(contains(preprocessedfiles,filename(1:end-4)))
          %  fprintf('SKIPPING File | ALREADY PREPROCESSED %s\n',...filename);
           % continue        
        %end
        %% Creating pre-processing log table
        preprocess_log  = table();
        preprocess_log.filename             = filename(1:end-4);
    
        %% Loading data
        EEG = pop_loadset(filename,eegfilepath);
        
        chanlist    = {EEG.chanlocs.labels}';
        EEG.chanlocs = accs_generateChanlocs(chanlist);
        srate       = EEG.srate;
        origEEG     = EEG;

        if ~contains(filename,'_WM_')
            epochs = eeg_regepochs(EEG,...
                'recurrence',2,'limits',[-1,1],...
                'eventtype','Marker');
        else
            epochs = pop_epoch(EEG,...
                'S 12',...
                [-1,1]);
        end
        
        %% Filtering 
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',120);
        EEG = pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1);

        %% Remove bad channels
        EEGdata      = EEG.data;
        noisyOut     = abrl_cleanEEG(EEGdata,srate,EEG.chanlocs,4);
        badchan_indx = noisyOut.noisyChannels.all;
        EEG          = pop_select(EEG,'nochannel',badchan_indx);
        preprocess_log.n_badchans           = length(badchan_indx);
    
        %% Eye blink & ECG removal
        EEGdata = EEG.data;            
        [clean_EEGdata,icainfo] = accs_eyeblinkremoval(...
            EEGdata,srate,EEG.chanlocs,'Fp1',20,50,[0.99,0.60],1,1);
        EEG.data = clean_EEGdata;
        preprocess_log.ncompremoved         = icainfo.ncompremoved;
        preprocess_log.signalslost          = icainfo.signalslost;
    
        %% Further ICA based cleaning
        if isempty(EEG.icaweights)         
            tempEEG = pop_clean_rawdata(EEG,'BurstRejection','on','BurstCriterion',15);
            tempEEG = pop_interp(tempEEG,EEG.chanlocs);  
            tempEEG = pop_runica(tempEEG, 'icatype', 'runica',...
                'extended',0,'sphering','off');
        
            % Add ica weights to main EEG data
            EEG.icawinv     = tempEEG.icawinv;
            EEG.icaweights  = tempEEG.icaweights;
            EEG.icachansind = tempEEG.icachansind;
            EEG.icasphere   = tempEEG.icasphere;
        end
        EEG = pop_iclabel(EEG, 'default');
        [EEG,icainfo] = accs_autoICAremoval(EEG,[0.99,0.60],1,...
            {'Muscle','Line Noise','Channel Noise'},...%{'Muscle','Eye','Heart','Line Noise','Channel Noise'}
            0.1,0.3);
        preprocess_log.ncompremoved         = preprocess_log.ncompremoved + icainfo.ncompremoved;
        preprocess_log.signalslost          = preprocess_log.signalslost + icainfo.signalslost;
    
        % Save file
        pop_saveset(EEG,...
            'filepath',outputDir,...
            'filename',[filename(1:end-4),'_ICA.set']);
    
        %% Interpolate bad channels
        EEG = pop_interp(EEG,origEEG.chanlocs);
        
        %% Epoching
        epochs = eeg_regepochs(EEG,...
           'recurrence',2,'limits',[-1,1],...
           'eventtype','Marker');
        epochdata = epochs.data;
        
        %% Epoch-level bad channel correction
        [epochdata,badchan] = accs_badchandetection(epochdata,1,chanlist);
    
        %% ASR
        state = accs_asr_calibrate(epochdata(:,:),srate,50);
        [clean_EEGdata,good] = accs_asr_epochcorrection(epochdata,state,10,50);
        epochs.data = clean_EEGdata;
    
        pop_saveset(epochs,...
            'filepath',outputDir,...
            'filename',[filename(1:end-4),'_ICA_ASR_epochs.set']);
    
        preprocess_log.asrdone              = mean(sum(good==0,2)/10);
        preprocess_log.time                 = datetime("now");
        writetable(preprocess_log,fullfile(outputDir,logfilename),'WriteMode','append');
    catch error
        continue
    end
end
