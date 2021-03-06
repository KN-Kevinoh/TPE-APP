# Generated by Django 2.2.16 on 2020-09-24 21:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=100, verbose_name='Email')),
                ('url_file', models.FileField(upload_to='media/dataset', verbose_name='Student video (URL)')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Creation date')),
            ],
            options={
                'verbose_name': 'dataSet',
            },
        ),
        migrations.CreateModel(
            name='Media',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url_file', models.FileField(upload_to='media/schedule', verbose_name='schedule video or image (URL)')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Creation date')),
            ],
            options={
                'verbose_name': 'media',
            },
        ),
        migrations.CreateModel(
            name='ModelML',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200, verbose_name='Model Name')),
                ('url_file', models.FileField(upload_to='media/models', verbose_name='Model (URL)')),
                ('description', models.TextField(blank=True, null=True, verbose_name='Describe Model')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='Creation date')),
                ('moified_at', models.DateTimeField(auto_now_add=True, verbose_name='Creation date')),
            ],
            options={
                'verbose_name': 'model ML',
            },
        ),
        migrations.CreateModel(
            name='Head',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url_file', models.FileField(upload_to='media/headPose', verbose_name='Cheat image (URL)')),
                ('head_left', models.BooleanField(default=False)),
                ('head_right', models.BooleanField(default=False)),
                ('nb_left', models.IntegerField(null=True, verbose_name='Number head left')),
                ('nb_right', models.IntegerField(null=True, verbose_name='Number head right')),
                ('media', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cheating.Media')),
            ],
            options={
                'verbose_name': 'head pose',
            },
        ),
        migrations.CreateModel(
            name='Emotion',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url_file', models.FileField(upload_to='media/emotions', verbose_name='Cheat image (URL)')),
                ('nb_sad', models.IntegerField(null=True, verbose_name='Number Sad')),
                ('nb_happy', models.IntegerField(null=True, verbose_name='Number Happy')),
                ('nb_disgust', models.IntegerField(null=True, verbose_name='Number Disgust')),
                ('nb_neutral', models.IntegerField(null=True, verbose_name='Number Neutral')),
                ('nb_fear', models.IntegerField(null=True, verbose_name='Number Fear')),
                ('nb_angry', models.IntegerField(null=True, verbose_name='Number Angry')),
                ('nb_surprise', models.IntegerField(null=True, verbose_name='Number Surprise')),
                ('media', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cheating.Media')),
            ],
            options={
                'verbose_name': 'emotion',
            },
        ),
        migrations.CreateModel(
            name='Cheat',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200, verbose_name='Model Name')),
                ('url_file', models.FileField(upload_to='media/cheats', verbose_name='Cheat image (URL)')),
                ('media', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='cheating.Media')),
            ],
            options={
                'verbose_name': 'cheat',
            },
        ),
    ]
