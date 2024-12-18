from django.shortcuts import render
from .forms import TranslationForm
from .translation.inference import translate_text


def translate_view(request):
    """Handle the translation form and display results."""
    translation_result = None

    if request.method == "POST":
        form = TranslationForm(request.POST)
        if form.is_valid():
            source_text = form.cleaned_data["source_text"]
            translation_result = translate_text(source_text)
    else:
        form = TranslationForm()

    return render(
        request,
        "translate.html",
        {"form": form, "translation_result": translation_result},
    )
