# Generated by Django 4.0.1 on 2022-03-11 08:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_rules'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='data',
            field=models.TextField(blank=True, null=True),
        ),
    ]
