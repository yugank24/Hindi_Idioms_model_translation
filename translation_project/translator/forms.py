from django import forms


class TranslationForm(forms.Form):
    source_text = forms.CharField(
        label="",
        widget=forms.Textarea(attrs={"rows": 2, "cols": 50}),
        required=True,
    )
