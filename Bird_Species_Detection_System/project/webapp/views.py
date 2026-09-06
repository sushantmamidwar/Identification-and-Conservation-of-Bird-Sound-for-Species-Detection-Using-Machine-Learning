import os
from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
# Create your views here.
from .models import *
from .forms import CreateUserForm
import numpy as np
import librosa
import pickle


# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'accounts/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'accounts/index.html')

@login_required(login_url='login')
def home2(request):
    return render(request, 'accounts/record.html')

@login_required(login_url='login')
def index(request):
	return render(request,"index.html")


with open("bird_sound_model_rf_main.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_main.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_main.pkl", "rb") as f:
    le = pickle.load(f)

@login_required(login_url='login')
def predictImage(request):
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    new_filename = "new_filename.mp3"
    filePathName = fs.save(new_filename, fileObj)
    test_image = "media/" + filePathName

    # -----------------------------
    # SAME FEATURE FUNCTION (NO CHANGE)
    # -----------------------------
    def extract_features(file_path):
        signal, sr = librosa.load(file_path, duration=5, sr=22050)

        target_length = sr * 5
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            signal = signal[:target_length]

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        mfcc_delta2_mean = np.mean(mfcc_delta2.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db.T, axis=0)
        mel_std = np.std(mel_db.T, axis=0)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal))

        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        harmonic = librosa.effects.harmonic(signal)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(signal))
        rms_mean = np.mean(librosa.feature.rms(y=signal))

        features = np.hstack([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta2_mean,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness],
            contrast_mean,
            tonnetz_mean,
            [zcr_mean, rms_mean]
        ])

        return features

    # -----------------------------
    # LOAD SAVED FILES
    # ----------------------------

    # -----------------------------
    # PREDICT SINGLE FILE
    # -----------------------------
    file_path =  test_image # change path

    features = extract_features(file_path)
    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_label = le.inverse_transform(prediction)

    print("Predicted Bird Class:", predicted_label[0])

    if predicted_label[0] == "Andean Guan":
        dict1 = {"name": "Andean Guan", "common_name": "Andean Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Andean_Guan.jpeg"}

    elif predicted_label[0] == "Andean Tinamou":
        dict1 = {"name": "Andean Tinamou", "common_name": "Andean Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Andean_Tinamou.jpeg"}

    elif predicted_label[0] == "Australian Brushturkey":
        dict1 = {"name": "Australian Brushturkey", "common_name": "Australian Brush-turkey", "place": "🇦🇺 Eastern & Northern Australia", "dir": "/media/BIRD_IMAGESS/Australian_Brushturkey.jpeg"}

    elif predicted_label[0] == "Band-tailed Guan":
        dict1 = {"name": "Band-tailed Guan", "common_name": "Band tailed Guan", "place": "🇧🇴 Bolivia, 🇦🇷 Argentina, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Band_tailed_Guan.jpeg"}

    elif predicted_label[0] == "Bartlett's Tinamou":
        dict1 = {"name": "Bartlett's Tinamou", "common_name": "Bartlett's Tinamou", "place": "🇧🇷 Western Brazil, 🇧🇴 Bolivia, 🇵🇪 Eastern Peru", "dir": "/media/BIRD_IMAGESS/Bartletts_Tinamou.jpeg"}

    elif predicted_label[0] == "Bearded Guan":
        dict1 = {"name": "Bearded Guan", "common_name": "Bearded Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Bearded_Guan.jpg"}

    elif predicted_label[0] == "Black-capped Tinamou":
        dict1 = {"name": "Black-capped Tinamou", "common_name": "Black-capped Tinamou", "place": "🇧🇷 Brazil, 🇨🇴 Colombia, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Black_capped_Tinamou.jpeg"}

    elif predicted_label[0] == "Blue-throated Piping Guan":
        dict1 = {"name": "Blue-throated Piping Guan", "common_name": "Blue-throated Piping-Guan", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Blue_throated_Piping_Guan.jpeg"}

    elif predicted_label[0] == "Brazilian Tinamou":
        dict1 = {"name": "Brazilian Tinamou", "common_name": "Brazilian Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Brazilian_Tinamou.jpeg"}

    elif predicted_label[0] == "Brown Tinamou":
        dict1 = {"name": "Brown Tinamou", "common_name": "Brown Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Brown_Tinamou.jpeg"}

    elif predicted_label[0] == "Brushland Tinamou":
        dict1 = {"name": "Brushland Tinamou", "common_name": "Brushland Tinamou", "place": "🇧🇴 Bolivia, 🇦🇷 Argentina, 🇵🇾 Paraguay", "dir": "/media/BIRD_IMAGESS/Brushland_Tinamou.jpeg"}

    elif predicted_label[0] == "Cauca Guan":
        dict1 = {"name": "Cauca Guan", "common_name": "Cauca Guan", "place": "🇨🇴 Colombia (Cauca Valley)", "dir": "/media/BIRD_IMAGESS/Cauca_Guan.jpeg"}

    elif predicted_label[0] == "Chaco Chachalaca":
        dict1 = {"name": "Chaco Chachalaca", "common_name": "Chaco Chachalaca", "place": "🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Chaco_Chachalaca.jpeg"}

    elif predicted_label[0] == "Chestnut-winged Chachalaca":
        dict1 = {"name": "Chestnut-winged Chachalaca", "common_name": "Chestnut-winged Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Chestnut_winged_Chachalaca.jpeg"}

    elif predicted_label[0] == "Cinereous Tinamou":
        dict1 = {"name": "Cinereous Tinamou", "common_name": "Cinereous Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia", "dir": "/media/BIRD_IMAGESS/Cinereous_Tinamou.jpeg"}

    elif predicted_label[0] == "Colombian Chachalaca":
        dict1 = {"name": "Colombian Chachalaca", "common_name": "Colombian Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Colombian_Chachalaca.jpeg"}

    elif predicted_label[0] == "Crested Guan":
        dict1 = {"name": "Crested Guan", "common_name": "Crested Guan", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇧🇿 Belize, 🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Crested_Guan.jpeg"}

    elif predicted_label[0] == "Dusky Megapode":
        dict1 = {"name": "Dusky Megapode", "common_name": "Dusky Megapode", "place": "🇮🇩 Indonesia (Maluku Islands)", "dir": "/media/BIRD_IMAGESS/Dusky_Megapode.jpeg"}

    elif predicted_label[0] == "Dusky-legged Guan":
        dict1 = {"name": "Dusky-legged Guan", "common_name": "Dusky-legged Guan", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/Dusky_legged_Guan.jpeg"}

    elif predicted_label[0] == "Dwarf Tinamou":
        dict1 = {"name": "Dwarf Tinamou", "common_name": "Dwarf Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Dwarf_Tinamou.jpeg"}

    elif predicted_label[0] == "Great Tinamou":
        dict1 = {"name": "Great Tinamou", "common_name": "Great Tinamou", "place": "🇲🇽 Mexico, 🇨🇴 Colombia, 🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Great_Tinamou.jpeg"}

    elif predicted_label[0] == "Grey Tinamou":
        dict1 = {"name": "Grey Tinamou", "common_name": "Gray Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Grey_Tinamou.jpeg"}

    elif predicted_label[0] == "Grey-headed Chachalaca":
        dict1 = {"name": "Grey-headed Chachalaca", "common_name": "Gray-headed Chachalaca", "place": "🇨🇴 Colombia, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Grey_headed_Chachalaca.jpeg"}

    elif predicted_label[0] == "Highland Tinamou":
        dict1 = {"name": "Highland Tinamou", "common_name": "Highland Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Highland_Tinamou.jpeg"}

    elif predicted_label[0] == "Little Chachalaca":
        dict1 = {"name": "Little Chachalaca", "common_name": "Little Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay", "dir": "/media/BIRD_IMAGESS/Little_Chachalaca.jpeg"}

    elif predicted_label[0] == "Little Tinamou":
        dict1 = {"name": "Little Tinamou", "common_name": "Little Tinamou", "place": "🇲🇽 Mexico, 🇨🇴 Colombia, 🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Little_Tinamou.jpeg"}

    elif predicted_label[0] == "Orange-footed Scrubfowl":
        dict1 = {"name": "Orange-footed Scrubfowl", "common_name": "Orange-footed Megapode", "place": "🇦🇺 Northern Australia, 🇮🇩 New Guinea", "dir": "/media/BIRD_IMAGESS/Orange_footed_Scrubfowl.jpeg"}

    elif predicted_label[0] == "Pale-browed Tinamou":
        dict1 = {"name": "Pale-browed Tinamou", "common_name": "Pale-browed Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Pale_browed_Tinamou.jpeg"}

    elif predicted_label[0] == "Plain Chachalaca":
        dict1 = {"name": "Plain Chachalaca", "common_name": "Plain Chachalaca", "place": "🇺🇸 Southern Texas, 🇲🇽 Mexico, 🇳🇮 Nicaragua, 🇨🇷 Costa Rica", "dir": "/media/BIRD_IMAGESS/Plain_Chachalaca.jpeg"}

    elif predicted_label[0] == "Red-legged Tinamou":
        dict1 = {"name": "Red-legged Tinamou", "common_name": "Red-legged Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil", "dir": "/media/BIRD_IMAGESS/Red_legged_Tinamou.jpeg"}

    elif predicted_label[0] == "Red-winged Tinamou":
        dict1 = {"name": "Red-winged Tinamou", "common_name": "Red-winged Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Red_winged_Tinamou.jpeg"}

    elif predicted_label[0] == "Rufous-bellied Chachalaca":
        dict1 = {"name": "Rufous-bellied Chachalaca", "common_name": "Rufous-bellied Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Rufous_bellied_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rufous-headed Chachalaca":
        dict1 = {"name": "Rufous-headed Chachalaca", "common_name": "Rufous-headed Chachalaca", "place": "🇨🇴 Colombia, 🇪🇨 Ecuador", "dir": "/media/BIRD_IMAGESS/Rufous_headed_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rufous-vented Chachalaca":
        dict1 = {"name": "Rufous-vented Chachalaca", "common_name": "Rufous-vented Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇹🇹 Trinidad & Tobago", "dir": "/media/BIRD_IMAGESS/Rufous_vented_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rusty-margined Guan":
        dict1 = {"name": "Rusty-margined Guan", "common_name": "Rusty-margined Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇷 Brazil", "dir": "/media/BIRD_IMAGESS/Rusty_margined_Guan.jpeg"}

    elif predicted_label[0] == "Slaty-breasted Tinamou":
        dict1 = {"name": "Slaty-breasted Tinamou", "common_name": "Slaty-breasted Tinamou", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇧🇿 Belize, 🇭🇳 Honduras, 🇳🇮 Nicaragua", "dir": "/media/BIRD_IMAGESS/Slaty_breasted_Tinamou.jpeg"}

    elif predicted_label[0] == "Small-billed Tinamou":
        dict1 = {"name": "Small-billed Tinamou", "common_name": "Small-billed Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Small_billed_Tinamou.jpeg"}

    elif predicted_label[0] == "Solitary Tinamou":
        dict1 = {"name": "Solitary Tinamou", "common_name": "Solitary Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Solitary_Tinamou.jpeg"}

    elif predicted_label[0] == "Speckled Chachalaca":
        dict1 = {"name": "Speckled Chachalaca", "common_name": "Speckled Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇧🇷 Brazil, 🇹🇹 Trinidad", "dir": "/media/BIRD_IMAGESS/Speckled_Chachalaca.jpeg"}

    elif predicted_label[0] == "Spix's Guan":
        dict1 = {"name": "Spix's Guan", "common_name": "Spix's Guan", "place": "🇧🇷 Brazil, 🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Spixs_Guan.jpeg"}

    elif predicted_label[0] == "Spotted Nothura":
        dict1 = {"name": "Spotted Nothura", "common_name": "Spotted Nothura", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/Spotted_Nothura.jpeg"}

    elif predicted_label[0] == "Tataupa Tinamou":
        dict1 = {"name": "Tataupa Tinamou", "common_name": "Tataupa Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Tataupa_Tinamou.jpeg"}

    elif predicted_label[0] == "Tawny-breasted Tinamou":
        dict1 = {"name": "Tawny-breasted Tinamou", "common_name": "Tawny-breasted Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Tawny_breasted_Tinamou.jpeg"}

    elif predicted_label[0] == "Thicket Tinamou":
        dict1 = {"name": "Thicket Tinamou", "common_name": "Thicket Tinamou", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇭🇳 Honduras, 🇳🇮 Nicaragua, 🇨🇷 Costa Rica", "dir": "/media/BIRD_IMAGESS/Thicket_Tinamou.jpeg"}

    elif predicted_label[0] == "Undulated Tinamou":
        dict1 = {"name": "Undulated Tinamou", "common_name": "Undulated Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Undulated_Tinamou.jpeg"}

    elif predicted_label[0] == "Variegated Tinamou":
        dict1 = {"name": "Variegated Tinamou", "common_name": "Variegated Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia", "dir": "/media/BIRD_IMAGESS/Variegated_Tinamou.jpeg"}

    elif predicted_label[0] == "West Mexican Chachalaca":
        dict1 = {"name": "West Mexican Chachalaca", "common_name": "West Mexican Chachalaca", "place": "🇲🇽 Western Mexico (Pacific slope)", "dir": "/media/BIRD_IMAGESS/West_Mexican_Chachalaca.jpeg"}

    elif predicted_label[0] == "White-bellied Nothura":
        dict1 = {"name": "White-bellied Nothura", "common_name": "White-bellied Nothura", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/White_bellied_Nothura.jpeg"}

    elif predicted_label[0] == "White-throated Tinamou":
        dict1 = {"name": "White-throated Tinamou", "common_name": "White-throated Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/White_throated_Tinamou.jpeg"}

    elif predicted_label[0] == "Yellow-legged Tinamou":
        dict1 = {"name": "Yellow-legged Tinamou", "common_name": "Yellow-legged Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Yellow_legged_Tinamou.jpeg"}

    else:
        dict1 = {"name": predicted_label[0], "common_name": predicted_label[0], "place": "Data not available", "dir": "/static/css/images/Clay_2.jpg"}

    return render(request, 'accounts/Result.html', {"bird": dict1})
   


@login_required(login_url='login')
def predictImage1(request):
    import sounddevice as sd
    from scipy.io.wavfile import write

    # Settings
    duration = 5  # seconds
    sample_rate = 44100  # Hz

    print("Recording started...")
    recording = sd.rec(int(duration * sample_rate), 
                    samplerate=sample_rate, 
                    channels=1)  # 1 = mono, 2 = stereo

    sd.wait()  # Wait until recording is finished

    # Save file
    write("media/output.wav", sample_rate, recording)
    # fileObj = request.FILES["document"]
    # fs = FileSystemStorage()
    # new_filename = "new_filename.mp3"
    # filePathName = fs.save(new_filename, fileObj)
    # test_image = "media/" + filePathName

    # -----------------------------
    # SAME FEATURE FUNCTION (NO CHANGE)
    # -----------------------------
    def extract_features(file_path):
        signal, sr = librosa.load(file_path, duration=5, sr=22050)

        target_length = sr * 5
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            signal = signal[:target_length]

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        mfcc_delta2_mean = np.mean(mfcc_delta2.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db.T, axis=0)
        mel_std = np.std(mel_db.T, axis=0)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal))

        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        harmonic = librosa.effects.harmonic(signal)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(signal))
        rms_mean = np.mean(librosa.feature.rms(y=signal))

        features = np.hstack([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta2_mean,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness],
            contrast_mean,
            tonnetz_mean,
            [zcr_mean, rms_mean]
        ])

        return features

    # -----------------------------
    # LOAD SAVED FILES
    # ----------------------------

    # -----------------------------
    # PREDICT SINGLE FILE
    # -----------------------------
    file_path =  r'media/output.wav' # change path
    # signal, sr = librosa.load(file_path, sr=22050)

    features = extract_features(file_path)
    features = features.reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_label = le.inverse_transform(prediction)

    print("Predicted Bird Class:", predicted_label[0])

    if predicted_label[0] == "Andean Guan":
        dict1 = {"name": "Andean Guan", "common_name": "Andean Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Andean_Guan.jpeg"}

    elif predicted_label[0] == "Andean Tinamou":
        dict1 = {"name": "Andean Tinamou", "common_name": "Andean Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Andean_Tinamou.jpeg"}

    elif predicted_label[0] == "Australian Brushturkey":
        dict1 = {"name": "Australian Brushturkey", "common_name": "Australian Brush-turkey", "place": "🇦🇺 Eastern & Northern Australia", "dir": "/media/BIRD_IMAGESS/Australian_Brushturkey.jpeg"}

    elif predicted_label[0] == "Band-tailed Guan":
        dict1 = {"name": "Band-tailed Guan", "common_name": "Band-tailed Guan", "place": "🇧🇴 Bolivia, 🇦🇷 Argentina, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Band_tailed_Guan.jpeg"}

    elif predicted_label[0] == "Bartlett's Tinamou":
        dict1 = {"name": "Bartlett's Tinamou", "common_name": "Bartlett's Tinamou", "place": "🇧🇷 Western Brazil, 🇧🇴 Bolivia, 🇵🇪 Eastern Peru", "dir": "/media/BIRD_IMAGESS/Bartletts_Tinamou.jpeg"}

    elif predicted_label[0] == "Bearded Guan":
        dict1 = {"name": "Bearded Guan", "common_name": "Bearded Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Bearded_Guan.jpeg"}

    elif predicted_label[0] == "Black-capped Tinamou":
        dict1 = {"name": "Black-capped Tinamou", "common_name": "Black-capped Tinamou", "place": "🇧🇷 Brazil, 🇨🇴 Colombia, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Black_capped_Tinamou.jpeg"}

    elif predicted_label[0] == "Blue-throated Piping Guan":
        dict1 = {"name": "Blue-throated Piping Guan", "common_name": "Blue-throated Piping-Guan", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Blue_throated_Piping_Guan.jpeg"}

    elif predicted_label[0] == "Brazilian Tinamou":
        dict1 = {"name": "Brazilian Tinamou", "common_name": "Brazilian Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Brazilian_Tinamou.jpeg"}

    elif predicted_label[0] == "Brown Tinamou":
        dict1 = {"name": "Brown Tinamou", "common_name": "Brown Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Brown_Tinamou.jpeg"}

    elif predicted_label[0] == "Brushland Tinamou":
        dict1 = {"name": "Brushland Tinamou", "common_name": "Brushland Tinamou", "place": "🇧🇴 Bolivia, 🇦🇷 Argentina, 🇵🇾 Paraguay", "dir": "/media/BIRD_IMAGESS/Brushland_Tinamou.jpeg"}

    elif predicted_label[0] == "Cauca Guan":
        dict1 = {"name": "Cauca Guan", "common_name": "Cauca Guan", "place": "🇨🇴 Colombia (Cauca Valley)", "dir": "/media/BIRD_IMAGESS/Cauca_Guan.jpeg"}

    elif predicted_label[0] == "Chaco Chachalaca":
        dict1 = {"name": "Chaco Chachalaca", "common_name": "Chaco Chachalaca", "place": "🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Chaco_Chachalaca.jpeg"}

    elif predicted_label[0] == "Chestnut-winged Chachalaca":
        dict1 = {"name": "Chestnut-winged Chachalaca", "common_name": "Chestnut-winged Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Chestnut_winged_Chachalaca.jpeg"}

    elif predicted_label[0] == "Cinereous Tinamou":
        dict1 = {"name": "Cinereous Tinamou", "common_name": "Cinereous Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia", "dir": "/media/BIRD_IMAGESS/Cinereous_Tinamou.jpeg"}

    elif predicted_label[0] == "Colombian Chachalaca":
        dict1 = {"name": "Colombian Chachalaca", "common_name": "Colombian Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Colombian_Chachalaca.jpeg"}

    elif predicted_label[0] == "Crested Guan":
        dict1 = {"name": "Crested Guan", "common_name": "Crested Guan", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇧🇿 Belize, 🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Crested_Guan.jpeg"}

    elif predicted_label[0] == "Dusky Megapode":
        dict1 = {"name": "Dusky Megapode", "common_name": "Dusky Megapode", "place": "🇮🇩 Indonesia (Maluku Islands)", "dir": "/media/BIRD_IMAGESS/Dusky_Megapode.jpeg"}

    elif predicted_label[0] == "Dusky-legged Guan":
        dict1 = {"name": "Dusky-legged Guan", "common_name": "Dusky-legged Guan", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/Dusky_legged_Guan.jpeg"}

    elif predicted_label[0] == "Dwarf Tinamou":
        dict1 = {"name": "Dwarf Tinamou", "common_name": "Dwarf Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Dwarf_Tinamou.jpeg"}

    elif predicted_label[0] == "Great Tinamou":
        dict1 = {"name": "Great Tinamou", "common_name": "Great Tinamou", "place": "🇲🇽 Mexico, 🇨🇴 Colombia, 🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Great_Tinamou.jpeg"}

    elif predicted_label[0] == "Grey Tinamou":
        dict1 = {"name": "Grey Tinamou", "common_name": "Gray Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Grey_Tinamou.jpeg"}

    elif predicted_label[0] == "Grey-headed Chachalaca":
        dict1 = {"name": "Grey-headed Chachalaca", "common_name": "Gray-headed Chachalaca", "place": "🇨🇴 Colombia, 🇪🇨 Ecuador, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Grey_headed_Chachalaca.jpeg"}

    elif predicted_label[0] == "Highland Tinamou":
        dict1 = {"name": "Highland Tinamou", "common_name": "Highland Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Highland_Tinamou.jpeg"}

    elif predicted_label[0] == "Little Chachalaca":
        dict1 = {"name": "Little Chachalaca", "common_name": "Little Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay", "dir": "/media/BIRD_IMAGESS/Little_Chachalaca.jpeg"}

    elif predicted_label[0] == "Little Tinamou":
        dict1 = {"name": "Little Tinamou", "common_name": "Little Tinamou", "place": "🇲🇽 Mexico, 🇨🇴 Colombia, 🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru", "dir": "/media/BIRD_IMAGESS/Little_Tinamou.jpeg"}

    elif predicted_label[0] == "Orange-footed Scrubfowl":
        dict1 = {"name": "Orange-footed Scrubfowl", "common_name": "Orange-footed Megapode", "place": "🇦🇺 Northern Australia, 🇮🇩 New Guinea", "dir": "/media/BIRD_IMAGESS/Orange_footed_Scrubfowl.jpeg"}

    elif predicted_label[0] == "Pale-browed Tinamou":
        dict1 = {"name": "Pale-browed Tinamou", "common_name": "Pale-browed Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Pale_browed_Tinamou.jpeg"}

    elif predicted_label[0] == "Plain Chachalaca":
        dict1 = {"name": "Plain Chachalaca", "common_name": "Plain Chachalaca", "place": "🇺🇸 Southern Texas, 🇲🇽 Mexico, 🇳🇮 Nicaragua, 🇨🇷 Costa Rica", "dir": "/media/BIRD_IMAGESS/Plain_Chachalaca.jpeg"}

    elif predicted_label[0] == "Red-legged Tinamou":
        dict1 = {"name": "Red-legged Tinamou", "common_name": "Red-legged Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇷 Brazil", "dir": "/media/BIRD_IMAGESS/Red_legged_Tinamou.jpeg"}

    elif predicted_label[0] == "Red-winged Tinamou":
        dict1 = {"name": "Red-winged Tinamou", "common_name": "Red-winged Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Red_winged_Tinamou.jpeg"}

    elif predicted_label[0] == "Rufous-bellied Chachalaca":
        dict1 = {"name": "Rufous-bellied Chachalaca", "common_name": "Rufous-bellied Chachalaca", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Rufous_bellied_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rufous-headed Chachalaca":
        dict1 = {"name": "Rufous-headed Chachalaca", "common_name": "Rufous-headed Chachalaca", "place": "🇨🇴 Colombia, 🇪🇨 Ecuador", "dir": "/media/BIRD_IMAGESS/Rufous_headed_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rufous-vented Chachalaca":
        dict1 = {"name": "Rufous-vented Chachalaca", "common_name": "Rufous-vented Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇹🇹 Trinidad & Tobago", "dir": "/media/BIRD_IMAGESS/Rufous_vented_Chachalaca.jpeg"}

    elif predicted_label[0] == "Rusty-margined Guan":
        dict1 = {"name": "Rusty-margined Guan", "common_name": "Rusty-margined Guan", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇷 Brazil", "dir": "/media/BIRD_IMAGESS/Rusty_margined_Guan.jpeg"}

    elif predicted_label[0] == "Slaty-breasted Tinamou":
        dict1 = {"name": "Slaty-breasted Tinamou", "common_name": "Slaty-breasted Tinamou", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇧🇿 Belize, 🇭🇳 Honduras, 🇳🇮 Nicaragua", "dir": "/media/BIRD_IMAGESS/Slaty_breasted_Tinamou.jpeg"}

    elif predicted_label[0] == "Small-billed Tinamou":
        dict1 = {"name": "Small-billed Tinamou", "common_name": "Small-billed Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Small_billed_Tinamou.jpeg"}

    elif predicted_label[0] == "Solitary Tinamou":
        dict1 = {"name": "Solitary Tinamou", "common_name": "Solitary Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Solitary_Tinamou.jpeg"}

    elif predicted_label[0] == "Speckled Chachalaca":
        dict1 = {"name": "Speckled Chachalaca", "common_name": "Speckled Chachalaca", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇧🇷 Brazil, 🇹🇹 Trinidad", "dir": "/media/BIRD_IMAGESS/Speckled_Chachalaca.jpeg"}

    elif predicted_label[0] == "Spix's Guan":
        dict1 = {"name": "Spix's Guan", "common_name": "Spix's Guan", "place": "🇧🇷 Brazil, 🇨🇴 Colombia, 🇻🇪 Venezuela, 🇬🇾 Guyana, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Spixs_Guan.jpeg"}

    elif predicted_label[0] == "Spotted Nothura":
        dict1 = {"name": "Spotted Nothura", "common_name": "Spotted Nothura", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/Spotted_Nothura.jpeg"}

    elif predicted_label[0] == "Tataupa Tinamou":
        dict1 = {"name": "Tataupa Tinamou", "common_name": "Tataupa Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Tataupa_Tinamou.jpeg"}

    elif predicted_label[0] == "Tawny-breasted Tinamou":
        dict1 = {"name": "Tawny-breasted Tinamou", "common_name": "Tawny-breasted Tinamou", "place": "🇨🇴 Colombia, 🇻🇪 Venezuela, 🇪🇨 Ecuador, 🇵🇪 Peru, 🇧🇴 Bolivia", "dir": "/media/BIRD_IMAGESS/Tawny_breasted_Tinamou.jpeg"}

    elif predicted_label[0] == "Thicket Tinamou":
        dict1 = {"name": "Thicket Tinamou", "common_name": "Thicket Tinamou", "place": "🇲🇽 Mexico, 🇬🇹 Guatemala, 🇭🇳 Honduras, 🇳🇮 Nicaragua, 🇨🇷 Costa Rica", "dir": "/media/BIRD_IMAGESS/Thicket_Tinamou.jpeg"}

    elif predicted_label[0] == "Undulated Tinamou":
        dict1 = {"name": "Undulated Tinamou", "common_name": "Undulated Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia, 🇻🇪 Venezuela", "dir": "/media/BIRD_IMAGESS/Undulated_Tinamou.jpeg"}

    elif predicted_label[0] == "Variegated Tinamou":
        dict1 = {"name": "Variegated Tinamou", "common_name": "Variegated Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇪 Peru, 🇨🇴 Colombia", "dir": "/media/BIRD_IMAGESS/Variegated_Tinamou.jpeg"}

    elif predicted_label[0] == "West Mexican Chachalaca":
        dict1 = {"name": "West Mexican Chachalaca", "common_name": "West Mexican Chachalaca", "place": "🇲🇽 Western Mexico (Pacific slope)", "dir": "/media/BIRD_IMAGESS/West_Mexican_Chachalaca.jpeg"}

    elif predicted_label[0] == "White-bellied Nothura":
        dict1 = {"name": "White-bellied Nothura", "common_name": "White-bellied Nothura", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/White_bellied_Nothura.jpeg"}

    elif predicted_label[0] == "White-throated Tinamou":
        dict1 = {"name": "White-throated Tinamou", "common_name": "White-throated Tinamou", "place": "🇧🇷 Brazil, 🇵🇾 Paraguay, 🇦🇷 Argentina, 🇺🇾 Uruguay", "dir": "/media/BIRD_IMAGESS/White_throated_Tinamou.jpeg"}

    elif predicted_label[0] == "Yellow-legged Tinamou":
        dict1 = {"name": "Yellow-legged Tinamou", "common_name": "Yellow-legged Tinamou", "place": "🇧🇷 Brazil, 🇧🇴 Bolivia, 🇵🇾 Paraguay, 🇦🇷 Argentina", "dir": "/media/BIRD_IMAGESS/Yellow_legged_Tinamou.jpeg"}

    else:
        dict1 = {"name": predicted_label[0], "common_name": predicted_label[0], "place": "Data not available", "dir": "/static/css/images/Clay_2.jpg"}

    return render(request, 'accounts/Result.html', {"bird": dict1})
   
    









