# Generated by Django 4.0.1 on 2022-02-23 10:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_segment_alter_sai_target_variable'),
    ]

    operations = [
        migrations.CreateModel(
            name='Rules',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('rule_data', models.TextField()),
            ],
            options={
                'db_table': 'Rules',
            },
        ),
    ]
